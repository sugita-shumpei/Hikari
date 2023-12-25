#include <hikari/assets/mitsuba/scene_importer.h>
#include <hikari/core/material.h>
#include <hikari/camera/perspective.h>
#include <hikari/camera/orthographic.h>
#include <hikari/film/hdr.h>
#include <hikari/film/spec.h>
#include <hikari/light/constant.h>
#include <hikari/light/envmap.h>
#include <hikari/light/area.h>
#include <hikari/shape/rectangle.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/cube.h>
#include <hikari/spectrum/srgb.h>
#include <hikari/spectrum/uniform.h>
#include <hikari/spectrum/srgb.h>
#include <hikari/texture/mipmap.h>
#include <hikari/texture/checkerboard.h>
#include <hikari/bsdf/diffuse.h>
#include <hikari/bsdf/conductor.h>
#include <hikari/bsdf/rough_conductor.h>
#include <hikari/bsdf/dielectric.h>
#include <hikari/bsdf/thin_dielectric.h>
#include <hikari/bsdf/rough_dielectric.h>
#include <hikari/bsdf/plastic.h>
#include <hikari/bsdf/rough_plastic.h>
#include <hikari/bsdf/phong.h>
#include <hikari/bsdf/mask.h>
#include <hikari/bsdf/bump_map.h>
#include <hikari/bsdf/normal_map.h>
#include <hikari/bsdf/two_sided.h>
#include <filesystem>
#include <unordered_set>
#include "xml_data.h"
#include "serialized_data.h"

struct hikari::MitsubaSceneImporter::Impl {
  std::filesystem::path  file_path = "";
  std::shared_ptr<Scene> scene     = nullptr;
  std::unordered_map<String, std::shared_ptr<MitsubaSerializedData>>  serialized_datas  = {};
  std::unordered_map<String, BsdfPtr   >                              ref_bsdfs         = {};//bsdf
  std::unordered_map<String, TexturePtr>                              ref_textures      = {};//texture
  std::vector<std::tuple<std::shared_ptr<ShapeMesh>, String, U32>>    meshes_serialized;
  std::vector<std::tuple<std::shared_ptr<ShapeMesh>, String>>         meshes_obj;
  std::vector<std::tuple<std::shared_ptr<ShapeMesh>, String>>         meshes_ply;
  // Spectrum を読み取る
  auto loadSpectrum(const MitsubaXMLData& xml_data, const MitsubaXMLSpectrum& spe_data) -> std::shared_ptr<Spectrum> { return nullptr; }
  auto loadPropSpectrum(const MitsubaXMLData& xml_data, const String& name, const MitsubaXMLProperties& properties) -> SpectrumPtr {
    auto rgb = getValueFromMap(properties.rgbs, name);
    if (rgb) { return SpectrumSrgb::create({ rgb->color.x, rgb->color.y, rgb->color.z }); }
    auto spec = getValueFromMap(properties.spectrums, name);
    if (spec) { return loadSpectrum(xml_data, *spec); }
    return nullptr;
  }
  // Texture  を読み取る, デフォルトでは有名テクスチャは読み込まず、参照テーブルの値を返す
  // replace_existing_object=trueのときのみ、参照テーブルの値を変更する
  auto loadTexture(const MitsubaXMLData& xml_data, const MitsubaXMLTexture&    tex_data, bool replace_existing_object = false) -> TexturePtr
  {
    using namespace std::string_literals;
    auto tex_obj = TexturePtr();
    if (tex_data.id != "") {
      tex_obj = getValueFromMap(ref_textures, tex_data.id, TexturePtr());
      if (!replace_existing_object && tex_obj) { return tex_obj; }
    }
    if (tex_data.type == "bitmap") {
      if (!tex_obj) { tex_obj = TextureMipmap::create(); }
      auto  texture_mipmap = tex_obj->convert<TextureMipmap>();

      auto filename = getValueFromMap(tex_data.properties.strings, "filename"s, ""s);
      if (filename!=""){ texture_mipmap->setFilename(filename); }

      auto filter_type_str = getValueFromMap(tex_data.properties.strings, "filter_type"s, ""s);
      if (filter_type_str == "bilinear") { texture_mipmap->setFilterType(TextureFilterType::eBilinear); }
      else if (filter_type_str == "nearest") { texture_mipmap->setFilterType(TextureFilterType::eNearest); }
      else if (filter_type_str == "") {}
      else { throw std::runtime_error("Failed To Parse filter_type!"); }

      auto wrap_mode_str = getValueFromMap(tex_data.properties.strings, "wrap_mode"s, ""s);
      if (wrap_mode_str == "repeat") { texture_mipmap->setWrapMode(TextureWrapMode::eRepeat); }
      else if (wrap_mode_str == "mirror") { texture_mipmap->setWrapMode(TextureWrapMode::eMirror); }
      else if (wrap_mode_str == "clamp") { texture_mipmap->setWrapMode(TextureWrapMode::eClamp); }
      else if (wrap_mode_str == "") {}
      else { throw std::runtime_error("Failed To Parse wrap_mode!"); }

      texture_mipmap->setRaw(getValueFromMap(tex_data.properties.booleans, "raw"s, false));
    }
    if (tex_data.type == "checkerboard") {
      if (!tex_obj) { tex_obj = TextureCheckerboard::create(); }
      auto texture_checkerboard = tex_obj->convert<TextureCheckerboard>();
      auto color0_ref = getValueFromMap(tex_data.properties.refs, "color0"s);
      auto color1_ref = getValueFromMap(tex_data.properties.refs, "color1"s);

      auto color0_xml_tex = getValueFromMap(tex_data.properties.textures, "color0"s, MitsubaXMLTexturePtr());
      auto color1_xml_tex = getValueFromMap(tex_data.properties.textures, "color1"s, MitsubaXMLTexturePtr());

      auto color0_rgb = getValueFromMap(tex_data.properties.rgbs, "color0"s);
      auto color1_rgb = getValueFromMap(tex_data.properties.rgbs, "color1"s);

      auto color0_spe = getValueFromMap(tex_data.properties.spectrums, "color0"s);
      auto color1_spe = getValueFromMap(tex_data.properties.spectrums, "color1"s);

      if (color0_ref) {
        auto tmp = getValueFromMap(xml_data.ref_textures, color0_ref->id, MitsubaXMLTexturePtr());
        if (tmp) color0_xml_tex = tmp;
      }
      if (color1_ref) {
        auto tmp = getValueFromMap(xml_data.ref_textures, color1_ref->id, MitsubaXMLTexturePtr());
        if (tmp) color1_xml_tex = tmp;
      }

      if (color0_xml_tex) {
        auto color_tex0 = loadTexture(xml_data, *color0_xml_tex);
        if (color_tex0 == nullptr) { throw std::runtime_error("Failed To Find Color0 Texture!"); }
        texture_checkerboard->setColor0(color_tex0);
      }
      if (color1_xml_tex) {
        auto color_tex1 = loadTexture(xml_data, *color1_xml_tex);
        if (color_tex1 == nullptr) { throw std::runtime_error("Failed To Find Color1 Texture!"); }
        texture_checkerboard->setColor1(color_tex1);
      }

      if (color0_rgb) {
        texture_checkerboard->setColor0(SpectrumSrgb::create({ color0_rgb->color.x,color0_rgb->color.y,color0_rgb->color.z }));
      }
      if (color1_rgb) {
        texture_checkerboard->setColor1(SpectrumSrgb::create({ color1_rgb->color.x,color1_rgb->color.y,color1_rgb->color.z }));
      }

      // TODO: spectrumのロードを行う
      // if (color0_spec)
      // if (color1_spec)
    }
    return tex_obj;
  }
  // Spectrum又はTextureを読み取る
  auto loadPropTexture(const MitsubaXMLData& xml_data, const String& name, const MitsubaXMLProperties& properties) -> TexturePtr {
    auto tex_xml = getValueFromMap(properties.textures, name, MitsubaXMLTexturePtr());
    if (tex_xml) {
      return loadTexture(xml_data, *tex_xml);
    }
    auto ref_xml = getValueFromMap(properties.refs, name, MitsubaXMLRef());
    if (ref_xml.id != "") {
      tex_xml = getValueFromMap(xml_data.ref_textures, ref_xml.id, MitsubaXMLTexturePtr());
      return loadTexture(xml_data, *tex_xml);
    }
    return nullptr;
  }
  auto loadPropSpectrumOrTexture(const MitsubaXMLData& xml_data, const String& name, const MitsubaXMLProperties& properties) -> std::optional<SpectrumOrTexture> {
    auto tex = loadPropTexture(xml_data, name, properties);
    if (tex) { return tex; }
    auto spe = loadPropSpectrum(xml_data, name, properties);
    if (spe) { return spe; }
    return std::nullopt;
  }
  // Texture又はFloatを読み取る
  auto loadPropFloatOrTexture(const MitsubaXMLData& xml_data, const String& name, const MitsubaXMLProperties& properties) -> std::optional<FloatOrTexture> {
    auto tex_xml = getValueFromMap(properties.textures, name, MitsubaXMLTexturePtr());
    if (tex_xml) {
      return loadTexture(xml_data, *tex_xml);
    }
    auto flt_val = getValueFromMap(properties.floats, name);
    if (flt_val) { return *flt_val; }
    return std::nullopt;
  }
  // Ref Texture  を読み取る ,
  void loadRefTextures(const MitsubaXMLData& xml_data) {
    using namespace std::string_literals;
    // まずテクスチャを一旦生成する
    for (auto& [ref,tex] : xml_data.ref_textures) {
      if (tex->type == "bitmap") {
        auto mipmap = TextureMipmap::create();
        ref_textures.insert({ ref,mipmap });
      }
      else if (tex->type == "checkerboard") {
        auto checkerboard = TextureCheckerboard::create();
        ref_textures.insert({ ref,checkerboard });
      }
      else {
        throw std::runtime_error("Failed Load Texture!");
      }
    }
    // 生成したテクスチャに値を設定する
    for (auto& [ref, tex] : xml_data.ref_textures) {
      loadTexture(xml_data, *tex, true);
    }
  }
  // Bsdfを読み取る, デフォルトでは有名テクスチャを読み込まず, 参照テーブルの値を返す.
  // replace_existing_object=trueのときのみ, 参照テーブルの値を変更する
  auto loadBsdf(const MitsubaXMLData& xml_data, const MitsubaXMLBsdfPtr& bsdf_data, bool replace_existing_object = false) -> BsdfPtr {
    std::unordered_map<String, BsdfDistributionType> str_to_dist = {
      {"beckman",BsdfDistributionType::eBeckman},
      {"ggx"    ,BsdfDistributionType::eGGX}
    };
    using namespace std::string_literals;
    auto mat_obj = BsdfPtr();
    if (bsdf_data->id != "") {
      mat_obj = getValueFromMap(ref_bsdfs, bsdf_data->id, BsdfPtr());
      if (!replace_existing_object && mat_obj) { return mat_obj; }
    }
    if (!mat_obj) {
      do {
        if (bsdf_data->type == "diffuse") {
          mat_obj = BsdfDiffuse::create(); break;
        }
        if (bsdf_data->type == "plastic") {
          mat_obj = BsdfPlastic::create(); break;
        }
        if (bsdf_data->type == "conductor") {
          mat_obj = BsdfConductor::create(); break;
        }
        if (bsdf_data->type == "dielectric") {
          mat_obj = BsdfDielectric::create(); break;
        }
        if (bsdf_data->type == "thindielectric") {
          mat_obj = BsdfThinDielectric::create(); break;
        }
        if (bsdf_data->type == "roughplastic") {
          mat_obj = BsdfRoughPlastic::create(); break;
        }
        if (bsdf_data->type == "roughconductor") {
          mat_obj = BsdfRoughConductor::create(); break;
        }
        if (bsdf_data->type == "roughdielectric") {
          mat_obj = BsdfRoughDielectric::create(); break;
        }
        if (bsdf_data->type == "phong") {
          mat_obj = BsdfPhong::create(); break;
        }
        if (bsdf_data->type == "mask") {
          mat_obj = BsdfMask::create(); break;
        }
        if (bsdf_data->type == "normalmap") {
          mat_obj = BsdfNormalMap::create(); break;
        }
        if (bsdf_data->type == "bumpmap") {
          mat_obj = BsdfBumpMap::create(); break;
        }
        if (bsdf_data->type == "twosided") {
          mat_obj = BsdfTwoSided::create(); break;
        }
        throw std::runtime_error("failed to support bsdf!");
      } while (false);
    }

    if (bsdf_data->type == "twosided" ) {
      auto& nested_bsdfs = bsdf_data->nested_bsdfs;
      if (nested_bsdfs.size() > 2 || nested_bsdfs.size() == 0) { throw std::runtime_error("number of nested bsdf must be 1 or 2!"); }
      if (nested_bsdfs.size() == 1) {
        auto mat0 = loadBsdf(xml_data, nested_bsdfs[0]);
        if (!mat0) { throw std::runtime_error("Failed To Load TwoSided Bsdf[0]!"); }
        mat_obj->convert<BsdfTwoSided>()->setBsdf(mat0);
      }
      if (nested_bsdfs.size() == 2) {
        auto mat0 = loadBsdf(xml_data, nested_bsdfs[0]);
        auto mat1 = loadBsdf(xml_data, nested_bsdfs[1]);
        if (!mat0) { throw std::runtime_error("Failed To Load TwoSided Bsdf[0]!"); }
        if (!mat1) { throw std::runtime_error("Failed To Load TwoSided Bsdf[1]!"); }
        mat_obj->convert<BsdfTwoSided>()->setBsdfs({mat0,mat1});
      }
      return mat_obj;
    }
    if (bsdf_data->type == "bumpmap"  ) {
      auto& nested_textures = bsdf_data->properties.nested_texs;
      if (nested_textures.size() != 1) { throw std::runtime_error("number of nested textures must be 1!"); }
      auto texture = loadTexture(xml_data, *nested_textures[0]);
      if (!texture) { throw std::runtime_error("Failed To Load BumpMap!"); }
      auto& nested_bsdfs = bsdf_data->nested_bsdfs;
      if (nested_bsdfs.size() != 1) { throw std::runtime_error("number of nested bsdf must be 1!"); }
      auto mat0 = loadBsdf(xml_data, nested_bsdfs[0]);
      if (!mat0) { throw std::runtime_error("Failed To Load BumpMap Bsdf!"); }

      auto bump = mat_obj->convert<BsdfBumpMap>();
      bump->setBsdf(mat0);
      bump->setTexture(texture);
      auto scale = getValueFromMap(bsdf_data->properties.floats, "scale"s, 1.0f);
      bump->setScale(scale);
      return mat_obj;
    }
    if (bsdf_data->type == "normalmap") {
      auto texture = loadPropTexture(xml_data, "normalmap"s, bsdf_data->properties);
      if (!texture) { throw std::runtime_error("Failed To Load NormalMap!"); }
      auto& nested_bsdfs = bsdf_data->nested_bsdfs;
      if (nested_bsdfs.size() != 1) { throw std::runtime_error("number of nested bsdf must be 1!"); }
      auto mat0 = loadBsdf(xml_data, nested_bsdfs[0]);
      if (!mat0) { throw std::runtime_error("Failed To Load BumpMap Bsdf!"); }

      auto norm = mat_obj->convert<BsdfNormalMap>();
      norm->setBsdf(mat0);
      norm->setTexture(texture);
      return mat_obj;
    }
    if (bsdf_data->type == "mask"     ) {
      auto opacity = loadPropSpectrumOrTexture(xml_data, "opacity"s, bsdf_data->properties);
      auto& nested_bsdfs = bsdf_data->nested_bsdfs;
      if (nested_bsdfs.size() != 1) { throw std::runtime_error("number of nested bsdf must be 1!"); }
      auto mat0 = loadBsdf(xml_data, nested_bsdfs[0]);
      if (!mat0) { throw std::runtime_error("Failed To Load BumpMap Bsdf!"); }

      auto mask = mat_obj->convert<BsdfMask>();
      mask->setBsdf(mat0);
      if (opacity) {
        mask->setOpacity(*opacity);
      }
      else {
        mask->setOpacity(SpectrumUniform::create(0.5f));
      }
      return mat_obj;
    }
    if (bsdf_data->type == "diffuse"  ) {
      auto reflectance = loadPropSpectrumOrTexture(xml_data, "reflectance"s, bsdf_data->properties);
      auto res = mat_obj->convert<BsdfDiffuse>();
      if (reflectance) { res->setReflectance(*reflectance); }
      return mat_obj;
    }
    if (bsdf_data->type == "conductor") {
      auto eta                  = loadPropSpectrumOrTexture(xml_data, "eta"s, bsdf_data->properties);
      auto k                    = loadPropSpectrumOrTexture(xml_data, "k"s  , bsdf_data->properties);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto res = mat_obj->convert<BsdfConductor>();
      if (eta) { res->setEta(*eta); }
      if (k) { res->setK(*k); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      return mat_obj;
    }
    if (bsdf_data->type == "roughconductor") {
      auto eta                  = loadPropSpectrumOrTexture(xml_data, "eta"s, bsdf_data->properties);
      auto k                    = loadPropSpectrumOrTexture(xml_data, "k"s  , bsdf_data->properties);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto alpha                = loadPropFloatOrTexture(xml_data, "alpha"s  , bsdf_data->properties);
      auto alpha_u              = loadPropFloatOrTexture(xml_data, "alpha_u"s, bsdf_data->properties);
      auto alpha_v              = loadPropFloatOrTexture(xml_data, "alpha_v"s, bsdf_data->properties);
      auto distribution         = getValueFromMap(str_to_dist,getValueFromMap(bsdf_data->properties.strings, "distribution"s, "beckman"s),BsdfDistributionType::eBeckman);

      auto res                  = mat_obj->convert<BsdfRoughConductor>();

      if (eta) { res->setEta(*eta); }
      if (k) { res->setK(*k); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      if (alpha) { res->setAlpha(*alpha); }
      if (alpha_u) { res->setAlphaU(*alpha_u); }
      if (alpha_v) { res->setAlphaV(*alpha_v); }
      res->setDistribution(distribution);
      return mat_obj;
    }
    if (bsdf_data->type == "dielectric") {
      auto int_ior                = getValueFromMap(bsdf_data->properties.floats,"int_ior"s);
      auto ext_ior                = getValueFromMap(bsdf_data->properties.floats,"ext_ior"s);
      auto specular_reflectance   = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto specular_transmittance = loadPropSpectrumOrTexture(xml_data, "specular_transmittance"s, bsdf_data->properties);
      auto res = mat_obj->convert<BsdfDielectric>();
      if (int_ior) { res->setIntIOR(*int_ior); }
      if (ext_ior) { res->setExtIOR(*ext_ior); }
      if (specular_reflectance  ) { res->setSpecularReflectance(*specular_reflectance); }
      if (specular_transmittance) { res->setSpecularTransmittance(*specular_transmittance); }
      return mat_obj;
    }
    if (bsdf_data->type == "thindielectric") {
      auto int_ior = getValueFromMap(bsdf_data->properties.floats, "int_ior"s);
      auto ext_ior = getValueFromMap(bsdf_data->properties.floats, "ext_ior"s);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto specular_transmittance = loadPropSpectrumOrTexture(xml_data, "specular_transmittance"s, bsdf_data->properties);
      auto res = mat_obj->convert<BsdfThinDielectric>();
      if (int_ior) { res->setIntIOR(*int_ior); }
      if (ext_ior) { res->setExtIOR(*ext_ior); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      if (specular_transmittance) { res->setSpecularTransmittance(*specular_transmittance); }
      return mat_obj;
    }
    if (bsdf_data->type == "roughdielectric") {
      auto int_ior = getValueFromMap(bsdf_data->properties.floats, "int_ior"s);
      auto ext_ior = getValueFromMap(bsdf_data->properties.floats, "ext_ior"s);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto specular_transmittance = loadPropSpectrumOrTexture(xml_data, "specular_transmittance"s, bsdf_data->properties);
      auto alpha = loadPropFloatOrTexture(xml_data, "alpha"s, bsdf_data->properties);
      auto alpha_u = loadPropFloatOrTexture(xml_data, "alpha_u"s, bsdf_data->properties);
      auto alpha_v = loadPropFloatOrTexture(xml_data, "alpha_v"s, bsdf_data->properties);
      auto distribution = getValueFromMap(str_to_dist, getValueFromMap(bsdf_data->properties.strings, "distribution"s, "beckman"s), BsdfDistributionType::eBeckman);
      auto res = mat_obj->convert<BsdfRoughDielectric>();
      if (int_ior) { res->setIntIOR(*int_ior); }
      if (ext_ior) { res->setExtIOR(*ext_ior); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      if (specular_transmittance) { res->setSpecularTransmittance(*specular_transmittance); }
      if (alpha) { res->setAlpha(*alpha); }
      if (alpha_u) { res->setAlphaU(*alpha_u); }
      if (alpha_v) { res->setAlphaV(*alpha_v); }
      res->setDistribution(distribution);
      return mat_obj;
    }
    if (bsdf_data->type == "plastic") {
      auto int_ior = getValueFromMap(bsdf_data->properties.floats, "int_ior"s);
      auto ext_ior = getValueFromMap(bsdf_data->properties.floats, "ext_ior"s);
      auto diffuse_reflectance = loadPropSpectrumOrTexture(xml_data, "diffuse_reflectance"s, bsdf_data->properties);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto nonlinear = getValueFromMap(bsdf_data->properties.booleans, "nonlinear"s, false);
      auto res = mat_obj->convert<BsdfPlastic>();
      if (int_ior) { res->setIntIOR(*int_ior); }
      if (ext_ior) { res->setExtIOR(*ext_ior); }
      if (diffuse_reflectance) { res->setDiffuseReflectance(*diffuse_reflectance); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      return mat_obj;
    }
    if (bsdf_data->type == "roughplastic") {
      auto int_ior = getValueFromMap(bsdf_data->properties.floats, "int_ior"s);
      auto ext_ior = getValueFromMap(bsdf_data->properties.floats, "ext_ior"s);
      auto diffuse_reflectance = loadPropSpectrumOrTexture(xml_data, "diffuse_reflectance"s, bsdf_data->properties);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto nonlinear = getValueFromMap(bsdf_data->properties.booleans, "nonlinear"s, false);
      auto alpha = loadPropFloatOrTexture(xml_data, "alpha"s, bsdf_data->properties);
      auto alpha_u = loadPropFloatOrTexture(xml_data, "alpha_u"s, bsdf_data->properties);
      auto alpha_v = loadPropFloatOrTexture(xml_data, "alpha_v"s, bsdf_data->properties);
      auto distribution = getValueFromMap(str_to_dist, getValueFromMap(bsdf_data->properties.strings, "distribution"s, "beckman"s), BsdfDistributionType::eBeckman);
      auto res = mat_obj->convert<BsdfRoughPlastic>();
      if (int_ior) { res->setIntIOR(*int_ior); }
      if (ext_ior) { res->setExtIOR(*ext_ior); }
      if (diffuse_reflectance) { res->setDiffuseReflectance(*diffuse_reflectance); }
      if (specular_reflectance) { res->setSpecularReflectance(*specular_reflectance); }
      if (alpha) { res->setAlpha(*alpha); }
      if (alpha_u) { res->setAlphaU(*alpha_u); }
      if (alpha_v) { res->setAlphaV(*alpha_v); }
      res->setDistribution(distribution);
      return mat_obj;
    }
    if (bsdf_data->type == "phong") {
      auto diffuse_reflectance  = loadPropSpectrumOrTexture(xml_data, "diffuse_reflectance"s, bsdf_data->properties);
      auto specular_reflectance = loadPropSpectrumOrTexture(xml_data, "specular_reflectance"s, bsdf_data->properties);
      auto exponent             = loadPropFloatOrTexture(xml_data, "exponent"s, bsdf_data->properties);
      auto res = mat_obj->convert<BsdfPhong>();
      if (diffuse_reflectance ) { res->setDiffuseReflectance(*diffuse_reflectance); }
      if (specular_reflectance) { res->setDiffuseReflectance(*specular_reflectance); }
      if (exponent) { res->setExponent(*exponent); }
      return mat_obj;
    }

    throw std::runtime_error("Unsupported BSDF Type!");

  }
  // Ref Bsdf     を読み取る
  void loadRefBsdfs(const MitsubaXMLData& xml_data) {
    using namespace std::string_literals;
    // まずMaterialを作成する.
    for (auto& [ref, bsdf_data] : xml_data.ref_bsdfs) {
      if (bsdf_data->type == "diffuse") {
        ref_bsdfs.insert({ ref,BsdfDiffuse::create() }); continue;
      }
      if (bsdf_data->type == "plastic") {
        ref_bsdfs.insert({ ref,BsdfPlastic::create() }); continue;
      }
      if (bsdf_data->type == "conductor") {
        ref_bsdfs.insert({ ref,BsdfConductor::create() }); continue;
      }
      if (bsdf_data->type == "dielectric") {
        ref_bsdfs.insert({ ref,BsdfDielectric::create() }); continue;
      }
      if (bsdf_data->type == "thindielectric") {
        ref_bsdfs.insert({ ref,BsdfThinDielectric::create() }); continue;
      }
      if (bsdf_data->type == "roughplastic") {
        ref_bsdfs.insert({ ref,BsdfRoughPlastic::create() }); continue;
      }
      if (bsdf_data->type == "roughconductor") {
        ref_bsdfs.insert({ ref,BsdfRoughConductor::create() }); continue;
      }
      if (bsdf_data->type == "roughdielectric") {
        ref_bsdfs.insert({ ref,BsdfRoughDielectric::create() }); continue;
      }
      if (bsdf_data->type == "phong") {
        ref_bsdfs.insert({ ref,BsdfPhong::create() }); continue;
      }
      if (bsdf_data->type == "mask") {
        ref_bsdfs.insert({ ref,BsdfMask::create() }); continue;
      }
      if (bsdf_data->type == "normalmap") {
        ref_bsdfs.insert({ ref,BsdfNormalMap::create() }); continue;
      }
      if (bsdf_data->type == "bumpmap") {
        ref_bsdfs.insert({ ref,BsdfBumpMap::create() }); continue;
      }
      if (bsdf_data->type == "twosided") {
        ref_bsdfs.insert({ ref,BsdfTwoSided::create() }); continue;
      }
      throw std::runtime_error("failed to load BSDF!");
    }
    for (auto& [ref, bsdf_data] : xml_data.ref_bsdfs) {
      loadBsdf(xml_data, bsdf_data, true);
    }
  }
  // Transformを読み取る(OK)
  auto loadTransform (const MitsubaXMLData& xml_data, const MitsubaXMLTransform& transform, const std::string& name) -> std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> {
    std::shared_ptr<Node> top_node = Node::create(name);
    constexpr size_t transform_idx_translate = 0;
    constexpr size_t transform_idx_rotate    = 1;
    constexpr size_t transform_idx_scale     = 2;
    constexpr size_t transform_idx_matrix    = 3;
    constexpr size_t transform_idx_lookat    = 4;
    auto   cur_node = top_node;
    for (auto& elem : transform.elements) {
      auto tmp_node = Node::create();
      switch (elem.data.index())
      {
      case transform_idx_translate:
      {
        hikari::TransformTRSData trs;
        trs.position = std::get<transform_idx_translate>(elem.data).value;
        tmp_node->setName("translate");
        tmp_node->setLocalTransform(trs);
        break;
      }
      case transform_idx_rotate:
      {
        hikari::TransformTRSData trs;
        auto& axis   = std::get<transform_idx_rotate>(elem.data).value;
        auto& angle  = std::get<transform_idx_rotate>(elem.data).angle;
        trs.rotation = glm::rotate(glm::identity<glm::quat>(), glm::radians(angle), axis);
        tmp_node->setName("rotate");
        tmp_node->setLocalTransform(trs);
        break;
      }
      case transform_idx_scale:
      {
        hikari::TransformTRSData trs;
        trs.scale = std::get<transform_idx_scale>(elem.data).value;
        tmp_node->setName("scale");
        tmp_node->setLocalTransform(trs);
        break;
      }
      case transform_idx_matrix:
      {
        hikari::TransformMatData mat;
        auto& values = std::get<transform_idx_matrix>(elem.data).values;
        if (values.size() == 9) {
          mat = Mat4x4(Mat3x3(
            Vec3(values[0], values[1], values[2]),
            Vec3(values[3], values[4], values[5]),
            Vec3(values[6], values[7], values[8])
          ));
          tmp_node->setName("mat3x3");
          tmp_node->setLocalTransform(mat);
          break;
        }
        if (values.size() == 16) {
          mat = Mat4x4(
            Vec4(values[0], values[1], values[2], values[3]),
            Vec4(values[4], values[5], values[6], values[7]),
            Vec4(values[8], values[9], values[10], values[11]),
            Vec4(values[12], values[13], values[14], values[15])
          );
          tmp_node->setName("mat4x4");
          tmp_node->setLocalTransform(mat);
          break;
        }
        return {nullptr,nullptr};
      }
      case transform_idx_lookat:
      {
        auto& origin = std::get<transform_idx_lookat>(elem.data).origin;
        auto& target = std::get<transform_idx_lookat>(elem.data).target;
        auto& up     = std::get<transform_idx_lookat>(elem.data).up;
        hikari::TransformMatData mat = glm::lookAt(origin,target,up);
        tmp_node->setName("lookat");
        tmp_node->setLocalTransform(mat);
        break;
      }
      break;
      default:
        return { nullptr,nullptr };
      }
      tmp_node->setParent(cur_node);
      cur_node = tmp_node;
    }
    return { top_node,cur_node };
  }
  // Envmap   が未解決
  auto loadRootLight (const MitsubaXMLData& xml_data, const MitsubaXMLEmitter & emi_data , const std::string& name) -> std::shared_ptr<Node> {
    auto head = std::shared_ptr<Node>();
    auto tail = std::shared_ptr<Node>();
    if (emi_data.to_world) {
      auto res = loadTransform(xml_data, emi_data.to_world.value(), name);
      head = res.first;
      tail = res.second;
    }
    else {
      auto node = Node::create(name);
      head = node;
      tail = node;
    }
    if (!head) { return nullptr; }

    auto light = [&xml_data,&emi_data,this]() -> LightPtr {
      if (emi_data.type == "constant") {
        auto res = LightConstant::create();
        {
          auto iter = emi_data.properties.rgbs.find("radiance");
          if (iter != emi_data.properties.rgbs.end()) {
            res->setRadiance(hikari::SpectrumSrgb::create({ iter->second.color.x,iter->second.color.y,iter->second.color.z }));
            return res;
          }
        }
        {
          auto iter = emi_data.properties.spectrums.find("radiance");
          if (iter != emi_data.properties.spectrums.end()) {
            auto spectrum = loadSpectrum(xml_data, iter->second);
            if (!spectrum) { return nullptr; }
            res->setRadiance(spectrum);
            return res;
          }
        }
        return nullptr;
      }
      if (emi_data.type == "envmap") {
        auto res  = LightEnvmap::create();
        {
          auto iter = emi_data.properties.strings.find("filename");
          if (iter == emi_data.properties.strings.end()) { return nullptr; }
          // TODO: テクスチャを読み取る
        }
        {
          auto iter = emi_data.properties.floats.find("scale");
          if (iter == emi_data.properties.floats.end()) {
            res->setScale(1.0f);
          }
          else {
            res->setScale(iter->second);
          }
        }
        return res;
      }
      return nullptr;
    }();
    if (!light) { return nullptr; }


    tail->setLight(light);
    return head;
  }
  // OK
  auto loadRootCamera(const MitsubaXMLData& xml_data, const MitsubaXMLSensor  & sen_data , const std::string& name) -> std::shared_ptr<Node> {
    auto head = std::shared_ptr<Node>();
    auto tail = std::shared_ptr<Node>();

    if (sen_data.to_world) {
      auto res = loadTransform(xml_data, sen_data.to_world.value(), name);
      head = res.first;
      tail = res.second;
    }
    else {
      auto node = Node::create(name);
      head = node;
      tail = node;
    }
    if (!head) { return nullptr; }

    auto camera = [&xml_data, &sen_data, this]()->CameraPtr {
      if (sen_data.type == "orthographic") {
        auto res = CameraOrthographic::create();
        auto iter_near_clip = sen_data.properties.floats.find("near_clip");
        auto iter_far_clip  = sen_data.properties.floats.find("far_clip");
        if (iter_near_clip!= sen_data.properties.floats.end()) { res->setNearClip(iter_near_clip->second); }
        else { res->setNearClip(1e-2); }
        if (iter_far_clip != sen_data.properties.floats.end()) { res->setFarClip(iter_far_clip->second)  ; }
        else { res->setFarClip(1e4); }
        return res;
      }
      if (sen_data.type == "perspective") {
        auto res = CameraPerspective::create();
        auto iter_fov          = sen_data.properties.floats.find("fov");
        auto iter_near_clip    = sen_data.properties.floats.find("near_clip");
        auto iter_far_clip     = sen_data.properties.floats.find("far_clip");
        auto iter_focal_length = sen_data.properties.strings.find("focal_length");
        auto iter_fov_axis     = sen_data.properties.strings.find("fov_axis");
        if (iter_fov != sen_data.properties.floats.end()) { res->setFov(iter_fov->second); }
        if ( iter_near_clip != sen_data.properties.floats.end()) { res->setNearClip(iter_near_clip->second); }
        else { res->setNearClip(1e-2); }
        if ( iter_far_clip != sen_data.properties.floats.end()) { res->setFarClip(iter_far_clip->second); }
        else { res->setFarClip(1e4); }
        if (iter_focal_length!=sen_data.properties.strings.end()){
          auto value_str = splitString(iter_focal_length->second, 'm');
          try {
            res->setFocalLength(std::stof(value_str[0]));
          }catch(std::invalid_argument&){}
          catch(std::out_of_range&){}
        }
        if (iter_fov_axis != sen_data.properties.strings.end()) {
          if (iter_fov_axis->second == "x") { res->setFovAxis(CameraFovAxis::eX); }
          if (iter_fov_axis->second == "y") { res->setFovAxis(CameraFovAxis::eY); }
          if (iter_fov_axis->second == "diagonal") { res->setFovAxis(CameraFovAxis::eDiagonal); }
          if (iter_fov_axis->second == "larger") { res->setFovAxis(CameraFovAxis::eLarger); }
          if (iter_fov_axis->second == "smaller") { res->setFovAxis(CameraFovAxis::eSmaller); }
        }
        return res;
      }
      return nullptr;
    }();
    if (!camera) { return nullptr; }

    auto& flm_data = sen_data.film;
    auto film   = [&xml_data, &flm_data, this]()->FilmPtr {
      if (flm_data.type == "hdrfilm") {
        auto res = FilmHdr::create();
        auto iter_width           = flm_data.properties.ingegers.find("width");
        auto iter_height          = flm_data.properties.ingegers.find("height");
        auto iter_pixel_format    = flm_data.properties.strings.find("pixel_format");
        auto iter_comonent_format = flm_data.properties.strings.find("comonent_format");
        if (iter_width != flm_data.properties.ingegers.end()) { res->setWidth(iter_width->second);  }
        else { res->setWidth(768);  }
        if (iter_height != flm_data.properties.ingegers.end()) { res->setHeight(iter_height->second); }
        else { res->setWidth(576); }
        if (iter_pixel_format != flm_data.properties.strings.end()) {
          if (iter_pixel_format->second == "luminance") { res->setPixelFormat(FilmPixelFormat::eLuminance); }
          if (iter_pixel_format->second == "luminance_alpha") { res->setPixelFormat(FilmPixelFormat::eLuminanceAlpha); }
          if (iter_pixel_format->second == "rgb") { res->setPixelFormat(FilmPixelFormat::eRGB); }
          if (iter_pixel_format->second == "rgba") { res->setPixelFormat(FilmPixelFormat::eRGBA); }
          if (iter_pixel_format->second == "xyz") { res->setPixelFormat(FilmPixelFormat::eXYZ); }
          if (iter_pixel_format->second == "xyza") { res->setPixelFormat(FilmPixelFormat::eXYZA); }
        }
        else {
          res->setPixelFormat(FilmPixelFormat::eRGB);
        }
        if (iter_comonent_format != flm_data.properties.strings.end()) {
          if (iter_comonent_format->second == "float16") { res->setComponentFormat(FilmComponentFormat::eFloat16); }
          if (iter_comonent_format->second == "float32") { res->setComponentFormat(FilmComponentFormat::eFloat32); }
          if (iter_comonent_format->second == "uint32" ) { res->setComponentFormat(FilmComponentFormat::eUInt32 ); }
        }
        else {
          res->setComponentFormat(FilmComponentFormat::eFloat16);
        }
        return res;
      }
      if (flm_data.type == "specfilm") {
        auto res = FilmSpec::create();
        auto iter_width = flm_data.properties.ingegers.find("width");
        auto iter_height = flm_data.properties.ingegers.find("height");
        auto iter_comonent_format = flm_data.properties.strings.find("comonent_format");
        if (iter_width != flm_data.properties.ingegers.end()) { res->setWidth(iter_width->second); }
        else { res->setWidth(768); }
        if (iter_height != flm_data.properties.ingegers.end()) { res->setHeight(iter_height->second); }
        else { res->setWidth(576); }
        if (iter_comonent_format != flm_data.properties.strings.end()) {
          if (iter_comonent_format->second == "float16") { res->setComponentFormat(FilmComponentFormat::eFloat16); }
          if (iter_comonent_format->second == "float32") { res->setComponentFormat(FilmComponentFormat::eFloat32); }
          if (iter_comonent_format->second == "uint32") { res->setComponentFormat(FilmComponentFormat::eUInt32); }
        }
        else {
          res->setComponentFormat(FilmComponentFormat::eFloat16);
        }
        return res;
      }
      return nullptr;
   }();
    if (!film) { return nullptr; }

    camera->setFilm(film);
    tail->setCamera(camera);
    return head;
  }
  // Instanceには対応しない
  auto loadRootShape(const MitsubaXMLData& xml_data, const MitsubaXMLShape& shp_data, const std::string& name) -> std::shared_ptr<Node> {
    using namespace std::string_literals;
    auto head = std::shared_ptr<Node>();
    auto tail = std::shared_ptr<Node>();

    if (shp_data.to_world) {
      auto res = loadTransform(xml_data, shp_data.to_world.value(), name);
      head = res.first;
      tail = res.second;
    }
    else {
      auto node = Node::create(name);
      head = node;
      tail = node;
    }
    if (!head) { return nullptr; }


    auto shape = [&xml_data, &shp_data, this]()->ShapePtr {
      auto flip_normals = getValueFromMap(shp_data.properties.booleans, "flip_normals"s, false);
      // OBJ
      if (shp_data.type == "obj") {
        auto res = hikari::ShapeMesh::create();
        auto filename = getValueFromMap(shp_data.properties.strings     , "filename"s, ""s);
        meshes_obj.push_back({ res,(std::filesystem::path(xml_data.filepath).parent_path() / filename).string() });
        res->setFaceNormals(getValueFromMap(shp_data.properties.booleans, "face_normals"s, false));
        res->setFlipUVs(getValueFromMap(shp_data.properties.booleans    , "flip_tex_coords"s, true));
        res->setFlipNormals(flip_normals);
        return res;
      }
      // PLY
      if (shp_data.type == "ply") {
        auto res = hikari::ShapeMesh::create();
        auto filename = getValueFromMap(shp_data.properties.strings, "filename"s, ""s);
        meshes_ply.push_back({ res,(std::filesystem::path(xml_data.filepath).parent_path() / filename).string() });
        res->setFaceNormals(getValueFromMap(shp_data.properties.booleans, "face_normals"s, false));
        res->setFlipUVs(getValueFromMap(shp_data.properties.booleans, "flip_tex_coords"s, false));
        res->setFlipNormals(flip_normals);
        return res;
      }
      // SERIALIZED
      if (shp_data.type == "serialized") {
        auto filename   = getValueFromMap(shp_data.properties.strings , "filename"s   ,""s);
        auto shape_idx  = getValueFromMap(shp_data.properties.ingegers, "shape_index"s,  0);
        if (shape_idx == 0) {
          shape_idx     = getValueFromMap(shp_data.properties.ingegers, "shapeIndex"s, 0);
        }
        auto  tmp_path        = (std::filesystem::path(xml_data.filepath).parent_path() / filename).string();
        auto& serialized_data = serialized_datas[tmp_path];
        {
          // もしserialized dataがみつからなければ再度読みだす
          if (!serialized_data) {
            serialized_data = std::shared_ptr<MitsubaSerializedData>(new MitsubaSerializedData());
            if (!serialized_data->load(tmp_path, MitsubaSerializedLoadType::eTemp)) { throw std::runtime_error("Failed To Load Mesh!"); }
          }
          // serialialized meshを読み取る
          if (!serialized_data->loadMesh(shape_idx)) { throw std::runtime_error("Failed To Load Serialied Data!"); }
        }
        auto res = hikari::ShapeMeshMitsubaSerialized::create(serialized_data->getMeshes()[shape_idx]);
        auto op_flip_normals    = getValueFromMap(shp_data.properties.booleans, "face_normals"s);
        auto op_flip_tex_coords = getValueFromMap(shp_data.properties.booleans, "flip_tex_coords"s);
        if (op_flip_normals) { res->setFlipNormals(*op_flip_normals); }
        if (op_flip_tex_coords) { res->setFlipUVs(*op_flip_tex_coords); }
        res->setFlipNormals(flip_normals);
        return res;
      }
       // 球体
       if (shp_data.type == "sphere") {
          auto res = hikari::ShapeSphere::create();
          res->setCenter(getValueFromMap(shp_data.properties.points, "center"s, { 0.0f,0.0f,0.0f }));
          res->setRadius(getValueFromMap(shp_data.properties.floats, "radius"s, 1.0f));
          res->setFlipNormals(flip_normals);
          return res;
        }
       // 立方体
       if (shp_data.type == "cube") {
          auto res = hikari::ShapeCube::create();
          res->setFlipNormals(flip_normals);
          return res;
        }
       // 四角
        if (shp_data.type == "rectangle") {
          auto res = hikari::ShapeRectangle::create();
          res->setFlipNormals(flip_normals);
          return res;
        }
        return nullptr;
    }();
    if (!shape) { return nullptr; }

    tail->setShape(shape);
    auto material = hikari::Material::create();
    if (shp_data.bsdf) {
      auto bsdf = loadBsdf(xml_data, shp_data.bsdf);
      if (bsdf) { material->setBsdf(bsdf); }
    }

    if (shp_data.emitter) {
      if (shp_data.emitter->type != "area") { throw std::runtime_error("Failed To Load Emitter!"); }
      auto area_light = hikari::LightArea::create();
      auto radiance = loadPropSpectrumOrTexture(xml_data, "radiance", shp_data.emitter->properties);
      if (radiance) { area_light->setRadiance(*radiance); }
      tail->setLight(area_light);
    }
    
    tail->setMaterial(material);
    return head;
  }
};

auto hikari::MitsubaSceneImporter::create() -> std::shared_ptr<MitsubaSceneImporter>
{
    return std::shared_ptr<MitsubaSceneImporter>(new MitsubaSceneImporter());
}

auto hikari::MitsubaSceneImporter::loadScene(const String& filename) -> std::shared_ptr<Scene>
{
    if (m_impl->file_path == filename){ return m_impl->scene; }
    tinyxml2::XMLDocument document;
    if (document.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS) { return nullptr; }
    hikari::MitsubaXMLParser parser = {};
    if (!document.Accept(&parser)) { return nullptr; }
    parser.setFilepath(filename);

    auto& xml_data      = parser.getData()     ;
    auto& sensor        = xml_data.sensor      ;
    auto& emitters      = xml_data.emitters    ;
    auto& shapes        = xml_data.shapes      ;
    auto& ref_bsdfs     = xml_data.ref_bsdfs   ;
    auto& ref_textures  = xml_data.ref_textures;
    auto  scene         = Scene::create("");

    // まず参照されている全てのTextureの参照を解決する
    m_impl->loadRefTextures(xml_data);
    // 次に参照されている全てのBSDF   の参照を解決する
    m_impl->loadRefBsdfs(xml_data);
    {
      auto node = m_impl->loadRootCamera(xml_data, sensor, "root_camera");
      if (!node) { return nullptr; }
      scene->addChild(node);
    }
    {
      size_t i = 0;
      for (auto& emitter : emitters) {
        std::string name = "root_lights[" + std::to_string(i) + "]";
        auto node = m_impl->loadRootLight(xml_data,*emitter, name);
        if (!node) { return nullptr; }
        scene->addChild(node);
        ++i;
      }
    }
    {
      size_t i = 0;
      for (auto& shape : shapes) {
        std::string name = "root_shapes[" + std::to_string(i) + "]";
        if (shape->id != "") { name = shape->id; }
        auto node = m_impl->loadRootShape(xml_data, *shape, name);
        if (!node) { return nullptr; }
        scene->addChild(node);
        ++i;
      }
    }
    return scene;
}

hikari::MitsubaSceneImporter::MitsubaSceneImporter():m_impl{new Impl()}
{}