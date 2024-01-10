#include <hikari/assets/mitsuba/scene_importer.h>
#include <hikari/camera/perspective.h>
#include <hikari/shape/mesh.h>
#include <hikari/light/envmap.h>
#include <hikari/core/node.h>
#include <hikari/core/film.h>
#include <hikari/core/material.h>
#include <hikari/core/surface.h>
#include <hikari/core/subsurface.h>
#include <hikari/core/bsdf.h>
#include <hikari/bsdf/diffuse.h>
#include <hikari/bsdf/plastic.h>
#include <hikari/texture/mipmap.h>
#include <hikari/texture/checkerboard.h>
#include <hikari/assets/image/exporter.h>
#include <hikari/spectrum/srgb.h>
#include <hikari/spectrum/uniform.h>
#include <gl_viewer.h>
#include <tonemap.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/LinearSpace.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <filesystem>
#include "deviceCode.h"
auto loadSpectrum(const hikari::SpectrumPtr& spectrum) -> owl::vec3f {
  owl::vec3f res = {};
  if (spectrum->getID() == hikari::SpectrumSrgb::ID()) {
    auto color = spectrum->convert<hikari::SpectrumSrgb>()->getColor();
    res.x = color.r;
    res.y = color.g;
    res.z = color.b;
  }
  if (spectrum->getID() == hikari::SpectrumUniform::ID()) {
    auto value = spectrum->convert<hikari::SpectrumUniform>()->getValue();
    res = owl::vec3f(value);
  }
  return res;
}
auto loadTexture(OWLContext context, const hikari::TexturePtr& texture, std::vector<OWLTexture>& objects) -> TextureData {
  TextureData data = {};
  if     (texture->getID() == hikari::TextureMipmap::ID()) {
    auto texture_mipmap = texture->convert<hikari::TextureMipmap>();
    auto mipmap         = texture_mipmap->getMipmap();
    auto bitmap         = mipmap->getImage(0);
    auto texel_format   = OWLTexelFormat{};
    auto width          = mipmap->getWidth ();
    auto height         = mipmap->getHeight();
    auto object         = OWLTexture(nullptr);
    if      (mipmap->getDataType() == hikari::MipmapDataType::eF16) {
      auto channel = mipmap->getChannel();
      auto image   = std::vector<hikari::F32>();
      if      (channel == 1) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_R32F;
        image = std::vector<hikari::F32>(width * height * 1);
        for (size_t i = 0; i < width * height; ++i) {
          image[1 * i + 0] = static_cast<const hikari::F16*>(bitmap->getData())[1 * i + 0];
        }
      }
      else if (channel == 2) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 4);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F16*>(bitmap->getData())[2 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F16*>(bitmap->getData())[2 * i + 1];
          image[4 * i + 2] = 0.0f;
          image[4 * i + 3] = 1.0f;
        }
      }
      else if (channel == 3) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 1);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F16*>(bitmap->getData())[3 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F16*>(bitmap->getData())[3 * i + 1];
          image[4 * i + 2] = static_cast<const hikari::F16*>(bitmap->getData())[3 * i + 2];
          image[4 * i + 3] = 1.0f;
        }
      }
      else if (channel == 4) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 4);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F16*>(bitmap->getData())[4 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F16*>(bitmap->getData())[4 * i + 1];
          image[4 * i + 2] = static_cast<const hikari::F16*>(bitmap->getData())[4 * i + 2];
          image[4 * i + 3] = static_cast<const hikari::F16*>(bitmap->getData())[4 * i + 3];
        }
      }
      object = owlTexture2DCreate(context, texel_format, width, height, image.data());
    }
    else if (mipmap->getDataType() == hikari::MipmapDataType::eF32) {
      auto channel = mipmap->getChannel();
      auto image = std::vector<hikari::F32>();
      if      (channel == 1) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_R32F;
        image = std::vector<hikari::F32>(width * height * 1);
        for (size_t i = 0; i < width * height; ++i) {
          image[1 * i + 0] = static_cast<const hikari::F32*>(bitmap->getData())[1 * i + 0];
        }
      }
      else if (channel == 2) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 4);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F32*>(bitmap->getData())[2 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F32*>(bitmap->getData())[2 * i + 1];
          image[4 * i + 2] = 0.0f;
          image[4 * i + 3] = 1.0f;
        }
      }
      else if (channel == 3) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 4);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F32*>(bitmap->getData())[3 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F32*>(bitmap->getData())[3 * i + 1];
          image[4 * i + 2] = static_cast<const hikari::F32*>(bitmap->getData())[3 * i + 2];
          image[4 * i + 3] = 1.0f;
        }
      }
      else if (channel == 4) {
        texel_format = OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F;
        image = std::vector<hikari::F32>(width * height * 4);
        for (size_t i = 0; i < width * height; ++i) {
          image[4 * i + 0] = static_cast<const hikari::F32*>(bitmap->getData())[4 * i + 0];
          image[4 * i + 1] = static_cast<const hikari::F32*>(bitmap->getData())[4 * i + 1];
          image[4 * i + 2] = static_cast<const hikari::F32*>(bitmap->getData())[4 * i + 2];
          image[4 * i + 3] = static_cast<const hikari::F32*>(bitmap->getData())[4 * i + 3];
        }
      }
      object = owlTexture2DCreate(context, texel_format, width, height, image.data());
    }

    auto uv_transform = glm::transpose(texture_mipmap->getUVTransform());
    objects.push_back(object);
    data.initObject(owlTextureGetObject(object, 0), reinterpret_cast<const owl::vec3f*>(&uv_transform[0]));
  }
  else if (texture->getID() == hikari::TextureCheckerboard::ID()) {
    auto texture_checkerboard = texture->convert<hikari::TextureCheckerboard>();
    auto color0 = texture_checkerboard->getColor0();
    auto color1 = texture_checkerboard->getColor1();
    auto color0_data = owl::vec3f();
    auto color1_data = owl::vec3f();
    auto uv_transform = glm::transpose(texture_checkerboard->getUVTransform());
    if      (color0.isSpectrum()) { color0_data = loadSpectrum(color0.getSpectrum());  }
    else if (color0.isTexture ()) {

    }
    if      (color1.isSpectrum()) { color1_data = loadSpectrum(color1.getSpectrum()); }
    else if (color1.isTexture ()) {

    }
    data.initChecker(color0_data, color1_data, reinterpret_cast<const owl::vec3f*>(&uv_transform[0]));
  }
  return data;
}

extern "C" char* deviceCode_ptx[];
int main() {
  auto context  = owlContextCreate();
  owlContextSetRayTypeCount(context, RAY_TYPE_COUNT);
  auto module   = owlModuleCreate(context, (const char*)deviceCode_ptx);
  auto viewer   = std::make_unique<hikari::test::owl::testlib::GLViewer>(owlContextGetStream(context, 0), 1024, 1024);
  auto importer = hikari::MitsubaSceneImporter::create();
  auto scene    = importer->load(R"(D:\Users\shums\Documents\C++\Hikari\data\mitsuba\matpreview\scene.xml)");
  auto camera   = scene->getCameras()[0];
  auto sensor_node = camera->getNode();
  auto lights  = scene->getLights();
  auto shapes  = scene->getShapes();
  auto texture_objects = std::vector<OWLTexture>();
  auto textures = std::vector<TextureData>();
  auto surfaces = std::vector<SurfaceData>();
  for (auto& shape : shapes) {
    auto surface_data = SurfaceData();
    auto material   = shape->getMaterial();
    auto surface    = material->getSurface();
    auto subsurface = surface->getSubSurface();
    auto bsdf       = subsurface->getBsdf();
    if (bsdf->getID() == hikari::BsdfDiffuse::ID()) {
      auto diffuse = bsdf->convert<hikari::BsdfDiffuse>();
      auto reflectance = diffuse->getReflectance();
      if      (reflectance.isSpectrum()) {
        auto refl_col = loadSpectrum(reflectance.getSpectrum());
        surface_data.initDiffuse(refl_col);
      }
      else if (reflectance.isTexture ()) {
        auto refl_tex = loadTexture(context, reflectance.getTexture(), texture_objects);
        surface_data.initDiffuse(textures.size());
        textures.push_back(refl_tex);
      }
      else    {}
    }
    if (bsdf->getID() == hikari::BsdfPlastic::ID()) {
      auto plastic          = bsdf->convert<hikari::BsdfPlastic>();

      auto diff_reflectance = plastic->getDiffuseReflectance();
      auto spec_reflectance = plastic->getSpecularReflectance();

      bool is_diff_reflectance_tex = diff_reflectance.isTexture();
      bool is_spec_reflectance_tex = spec_reflectance.isTexture();

      auto eta                             = plastic->getEta();
      auto int_fresnel_diffuse_reflectance = plastic->getIntFresnelDiffuseReflectance();
      if (!plastic->getNonLinear()) { int_fresnel_diffuse_reflectance *= -1.0f; }
      if (is_diff_reflectance_tex && is_spec_reflectance_tex) {
        auto diff_refl_tex = loadTexture(context, diff_reflectance.getTexture(), texture_objects);
        auto spec_refl_tex = loadTexture(context, spec_reflectance.getTexture(), texture_objects);
        surface_data.initPlastic(textures.size() + 0u, textures.size() + 1u, eta, int_fresnel_diffuse_reflectance);
        textures.push_back(diff_refl_tex);
        textures.push_back(spec_refl_tex);
      }
      else if (is_diff_reflectance_tex) {
        auto diff_refl_tex = loadTexture(context, diff_reflectance.getTexture(), texture_objects);
        auto spec_refl_col = loadSpectrum(spec_reflectance.getSpectrum());
        surface_data.initPlastic(textures.size() + 0u, spec_refl_col, eta, int_fresnel_diffuse_reflectance);
        textures.push_back(diff_refl_tex);
      }
      else if (is_spec_reflectance_tex) {
        auto spec_refl_tex = loadTexture(context, spec_reflectance.getTexture(), texture_objects);
        auto diff_refl_col = loadSpectrum(diff_reflectance.getSpectrum());
        surface_data.initPlastic(diff_refl_col, textures.size() + 0u, eta, int_fresnel_diffuse_reflectance);
        textures.push_back(spec_refl_tex);
      }
      else {
        auto diff_refl_col = loadSpectrum(diff_reflectance.getSpectrum());
        auto spec_refl_col = loadSpectrum(spec_reflectance.getSpectrum());
        surface_data.initPlastic(diff_refl_col, spec_refl_col, eta, int_fresnel_diffuse_reflectance);
      }

    }
    surfaces.push_back(surface_data);
  }

  auto texture_buffer = owlDeviceBufferCreate(context, (OWLDataType)(OWLDataType::OWL_USER_TYPE_BEGIN + sizeof(TextureData)), textures.size(), textures.data());
  auto surface_buffer = owlDeviceBufferCreate(context, (OWLDataType)(OWLDataType::OWL_USER_TYPE_BEGIN + sizeof(SurfaceData)), surfaces.size(), surfaces.data());
  // raygen
  OWLVarDecl vardecls_raygen[] = {
    {"camera.dir_u"    ,OWLDataType::OWL_FLOAT3     , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,dir_u)},
    {"camera.dir_v"    ,OWLDataType::OWL_FLOAT3     , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,dir_v)},
    {"camera.dir_w"    ,OWLDataType::OWL_FLOAT3     , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,dir_w)},
    {"camera.eye"      ,OWLDataType::OWL_FLOAT3     , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,eye)},
    {"camera.near_clip",OWLDataType::OWL_FLOAT      , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,near_clip)},
    {"camera.far_clip" ,OWLDataType::OWL_FLOAT      , offsetof(SBTRaygenData,camera)+ offsetof(CameraData,far_clip)},
    {"frame_buffer",OWLDataType::OWL_BUFPTR     , offsetof(SBTRaygenData,frame_buffer) },
    {"accum_buffer",OWLDataType::OWL_BUFPTR     , offsetof(SBTRaygenData,accum_buffer) },
    {"width"       ,OWLDataType::OWL_INT        , offsetof(SBTRaygenData,width)        },
    {"height"      ,OWLDataType::OWL_INT        , offsetof(SBTRaygenData,height)       },
    {"sample"      ,OWLDataType::OWL_INT        , offsetof(SBTRaygenData,sample)       },
    {nullptr}
  };
  auto raygen       = owlRayGenCreate(context, module, "default", sizeof(SBTRaygenData), vardecls_raygen, -1);
  auto accum_buffer = OWLBuffer(nullptr);
  auto frame_buffer = OWLBuffer(nullptr);
  {
    // CAMERA->WORLD(MITSUBA)
    auto view_matrix     = sensor_node->getGlobalTransform().getMat();
    view_matrix[0]      *= -1.0f;
    view_matrix[2]      *= -1.0f;
    // SCREEN->CAMERA
    auto proj_matrix     = camera->convert<hikari::CameraPerspective>()->getProjMatrix_Infinite();
    auto inv_proj_matrix = glm::inverse(hikari::Mat3x3(proj_matrix));
    auto view_matrix3    = hikari::Mat3x3(view_matrix);

    auto camera_eye      = hikari::Vec3(view_matrix[3]);
    view_matrix3         = view_matrix3 * inv_proj_matrix;
    //view_matrix[0]      *= ax;
    //view_matrix[1]      *= ay;
    //view_matrix[2]      *= az;

    auto film    = camera->getFilm();
    accum_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, film->getWidth() * film->getHeight(), nullptr);
    frame_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, film->getWidth() * film->getHeight(), nullptr);

    owlRayGenSetBuffer(raygen, "frame_buffer", frame_buffer);
    owlRayGenSetBuffer(raygen, "accum_buffer", accum_buffer);
    owlRayGenSet1i(raygen , "width"     , film->getWidth() );
    owlRayGenSet1i(raygen , "height"    , film->getHeight());
    owlRayGenSet1i(raygen , "sample"    , 0);
    owlRayGenSet3fv(raygen, "camera.dir_u", (const float*)&view_matrix3[0]);
    owlRayGenSet3fv(raygen, "camera.dir_v", (const float*)&view_matrix3[1]);
    owlRayGenSet3fv(raygen, "camera.dir_w", (const float*)&view_matrix3[2]);
    owlRayGenSet3fv(raygen, "camera.eye"  , (const float*)&camera_eye);
    owlRayGenSet1f(raygen , "camera.near_clip", camera->convert<hikari::CameraPerspective>()->getNearClip());
    owlRayGenSet1f(raygen , "camera.far_clip" , camera->convert<hikari::CameraPerspective>()->getFarClip());

  }
  // miss
  OWLVarDecl vardecls_miss[] = {
    {nullptr}
  };
  auto miss      = owlMissProgCreate(context, module, "default", sizeof(SBTMissData),vardecls_miss, -1);
  auto miss2     = owlMissProgCreate(context, module, "occlude", sizeof(SBTMissData), vardecls_miss, -1);
  {
  }
  // hitgroup
  OWLVarDecl vardecls_hitgroup[] = {
    {"vertex_buffer",OWLDataType::OWL_BUFPTR, offsetof(SBTHitgroupData,vertex_buffer) },
    {"normal_buffer",OWLDataType::OWL_BUFPTR, offsetof(SBTHitgroupData,normal_buffer) },
    {"texcrd_buffer",OWLDataType::OWL_BUFPTR, offsetof(SBTHitgroupData,texcrd_buffer) },
    { "index_buffer",OWLDataType::OWL_BUFPTR, offsetof(SBTHitgroupData, index_buffer) },
    { "surfaces",OWLDataType::OWL_USHORT, offsetof(SBTHitgroupData, surfaces) },
    {nullptr}
  };
  // geomtype
  auto geom_type = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOMETRY_TRIANGLES, sizeof(SBTHitgroupData), vardecls_hitgroup, -1);
  owlGeomTypeSetClosestHit(geom_type, RAY_TYPE_RADIANCE, module, "default_triangle");
  owlGeomTypeSetClosestHit(geom_type, RAY_TYPE_OCCLUDED, module, "occlude_triangle");
  // group
  auto group     = static_cast<OWLGroup>(nullptr);
  {
    size_t i = 0;
    std::vector<OWLGeom> geoms = {};
    for (auto& shape : shapes) {
      auto node      = shape->getNode();
      auto transform = node->getGlobalTransform().getMat();// local->world
      auto triangles = owlGeomCreate(context,geom_type);
      auto mesh      = shape->convert<hikari::ShapeMesh>();
      auto vertices  = mesh->getVertexPositions();
      auto normals   = mesh->getVertexNormals();
      auto uvs       = mesh->getVertexUVs();
      auto indices   = mesh->getFaces();
      for (auto& vertex : vertices) {
        auto vertex_ = transform * hikari::Vec4(vertex, 1.0f);
        vertex       = hikari::Vec3(vertex_) / vertex_.w;
      }
      for (auto& normal : normals) {
        auto normal_ = glm::transpose(glm::inverse(hikari::Mat3x3(transform))) * hikari::Vec3(normal);
        normal       = glm::normalize(normal_);
      }

      auto vertex_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, vertices.size(), vertices.data());
      auto normal_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3,  normals.size(), normals.data());
      auto texcrd_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT2, uvs.size(), uvs.data());
      auto  index_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_UINT3 , indices.size()/3, indices.data());

      auto geom = owlGeomCreate(context, geom_type);
      owlTrianglesSetVertices(geom, vertex_buffer, vertices.size(), sizeof(vertices[0]), 0);
      owlTrianglesSetIndices(geom, index_buffer, indices.size()/3, 3*sizeof(indices[0]), 0);
      owlGeomSetBuffer(geom, "vertex_buffer", vertex_buffer);
      owlGeomSetBuffer(geom, "normal_buffer", normal_buffer);
      owlGeomSetBuffer(geom, "texcrd_buffer", texcrd_buffer);
      owlGeomSetBuffer(geom,  "index_buffer",  index_buffer);
      owlGeomSet1us(geom, "surfaces", i);
      geoms.push_back(geom);
      ++i;
    }

    if (geoms.size() >= 1) {
      auto geom_group = owlTrianglesGeomGroupCreate(context, geoms.size(), geoms.data(), OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(geom_group);

      group = owlInstanceGroupCreate(context, 1, &geom_group);
      owlGroupBuildAccel(group);
    }
  }
  OWLVarDecl vardecls_params[] = {
    {"tlas"                    ,OWLDataType::OWL_GROUP , offsetof(LaunchParams,tlas)},
    {"surfaces"                ,OWLDataType::OWL_BUFPTR, offsetof(LaunchParams,surfaces)},
    {"textures"                ,OWLDataType::OWL_BUFPTR, offsetof(LaunchParams,textures)},
    {"light.envmap.to_local[0]",OWLDataType::OWL_FLOAT4, offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,to_local) + sizeof(owl::vec4f) * 0},
    {"light.envmap.to_local[1]",OWLDataType::OWL_FLOAT4, offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,to_local) + sizeof(owl::vec4f) * 1},
    {"light.envmap.to_local[2]",OWLDataType::OWL_FLOAT4, offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,to_local) + sizeof(owl::vec4f) * 2},
    {"light.envmap.to_local[3]",OWLDataType::OWL_FLOAT4, offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,to_local) + sizeof(owl::vec4f) * 3},
    {"light.envmap.texture"    ,OWLDataType::OWL_TEXTURE,offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,texture) },
    {"light.envmap.scale"      ,OWLDataType::OWL_FLOAT  ,offsetof(LaunchParams,light) + offsetof(LightData,envmap) + offsetof(LightEnvmapData,scale)   },
    {nullptr}
  };
  auto params    = owlParamsCreate(context, sizeof(LaunchParams), vardecls_params, -1);
  {
    owlParamsSetGroup(params,"tlas",group);

    {
      auto node = lights[0]->getNode();
      auto transform = glm::transpose(glm::inverse(node->getGlobalTransform().getMat()));
      auto envmap = lights[0]->convert<hikari::LightEnvmap>();
      auto bitmap = envmap->getBitmap();
      auto channel = bitmap->getChannel();

      owlParamsSetBuffer(params, "surfaces", surface_buffer);
      owlParamsSetBuffer(params, "textures", texture_buffer);
      owlParamsSet4fv(params, "light.envmap.to_local[0]", (const float*)&transform[0]);
      owlParamsSet4fv(params, "light.envmap.to_local[1]", (const float*)&transform[1]);
      owlParamsSet4fv(params, "light.envmap.to_local[2]", (const float*)&transform[2]);
      owlParamsSet4fv(params, "light.envmap.to_local[3]", (const float*)&transform[3]);
      owlParamsSet1f(params, "light.envmap.scale", envmap->getScale());

      if (channel > 1) {
        auto texdata = std::vector<std::array<hikari::F32, 4>>(bitmap->getWidth() * bitmap->getHeight());
        {
          for (size_t i = 0; i < texdata.size(); ++i) {
            texdata[i] = { 0.0f,0.0f,0.0f,1.0f };
            if (bitmap->getDataType() == hikari::BitmapDataType::eF16) {
              for (size_t j = 0; j < channel; ++j) {
                texdata[i][j] = ((const hikari::F16*)bitmap->getData())[channel * i + j];
              }
            }
            else if (bitmap->getDataType() == hikari::BitmapDataType::eF32) {
              for (size_t j = 0; j < channel; ++j) {
                texdata[i][j] = ((const hikari::F32*)bitmap->getData())[channel * i + j];
              }
            }
          }
        }
        auto texture = owlTexture2DCreate(context, OWLTexelFormat::OWL_TEXEL_FORMAT_RGBA32F, bitmap->getWidth(), bitmap->getHeight(), texdata.data(),OWLTextureFilterMode::OWL_TEXTURE_LINEAR);
        owlParamsSetTexture(params, "light.envmap.texture", texture);
      }
    }
  }

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context, (OWLBuildSBTFlags)(OWLBuildSBTFlags::OWL_SBT_ALL2));

  {

    auto film = camera->getFilm();
    auto tonemap = hikari::test::owl::testlib::Tonemap(film->getWidth(), film->getHeight(), 0.104f);
    tonemap.init();
    struct TracerData {
      hikari::CameraPtr                          camera;
      hikari::test::owl::testlib::Tonemap*       p_tonemap;
      OWLContext                                 context;
      OWLRayGen                                  raygen;
      OWLParams                                  params;
      OWLBuffer                                  accum_buffer;
      OWLBuffer                                  frame_buffer;
      int                                        accum_sample;
      bool                                       estimate_luminance;
      std::string                                screen_filename;
      bool                                       screen_shot;
    } tracer_data = {
        camera,&tonemap,context, raygen,params,accum_buffer,frame_buffer, 0,true,std::string(""),false
    };

    auto resize_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, int old_w, int old_h, int new_w, int        new_h) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      owlBufferResize(p_tracer_data->accum_buffer, new_w * new_h);
      owlBufferResize(p_tracer_data->frame_buffer, new_w * new_h);
      owlParamsSet2i(p_tracer_data->params, "frame_size", new_w, new_h);
      return true;
      };
    auto presskey_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, hikari::test::owl::testlib::KeyType           key) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      //if (key == hikari::test::owl::testlib::KeyType::eW) { p_tracer_data->p_camera->processPressKeyW(1.0f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eS) { p_tracer_data->p_camera->processPressKeyS(1.0f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eA) { p_tracer_data->p_camera->processPressKeyA(1.0f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eD) { p_tracer_data->p_camera->processPressKeyD(1.0f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eLeft) { p_tracer_data->p_camera->processPressKeyLeft(0.5f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eRight) { p_tracer_data->p_camera->processPressKeyRight(0.5f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eUp) { p_tracer_data->p_camera->processPressKeyUp(0.5f); return true; }
      //if (key == hikari::test::owl::testlib::KeyType::eDown) { p_tracer_data->p_camera->processPressKeyDown(0.5f); return true; }
      return false;
      };
    auto press_mouse_button_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, hikari::test::owl::testlib::MouseButtonType mouse) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      if (mouse == hikari::test::owl::testlib::MouseButtonType::eLeft) {
        int width; int height;
        double x; double y;
        p_viewer->getCursorPosition(x, y);
        p_viewer->getWindowSize(width, height);
        // ���オ(0,0), �E����(1,1)
        float sx = std::clamp((float)x / (float)width, 0.0f, 1.0f);
        float sy = std::clamp((float)y / (float)height, 0.0f, 1.0f);
        printf("%f %f\n", sx, sy);
        //if (sx < 0.5f) { p_tracer_data->p_camera->processPressKeyLeft(0.5f - sx); }
        //else { p_tracer_data->p_camera->processPressKeyRight(sx - 0.5f); }
        //if (sy < 0.5f) { p_tracer_data->p_camera->processPressKeyUp(0.5f - sy); }
        //else { p_tracer_data->p_camera->processPressKeyDown(sy - 0.5f); }
        return true;
      }
      return false;
      };
    auto mouse_scroll_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, double x, double y) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      if (y != 0.0f) {
        //p_tracer_data->p_camera->processMouseScrollY(y);
        return true;
      }
      return false;
      };
    auto update_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer) {
      };
    auto render_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, void* p_fb_data) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      auto film = p_tracer_data->camera->getFilm();
      owlRayGenSet1i(p_tracer_data->raygen, "sample", p_tracer_data->accum_sample);
      owlBuildSBT(p_tracer_data->context, (OWLBuildSBTFlags)(OWL_SBT_RAYGENS | OWL_SBT_CALLABLES));
      owlLaunch2D(p_tracer_data->raygen, film->getWidth(), film->getHeight(), p_tracer_data->params);
      p_tracer_data->p_tonemap->launch(owlContextGetStream(p_tracer_data->context, 0),
        (const float3*)owlBufferGetPointer(p_tracer_data->frame_buffer, 0),
        (unsigned int*)p_fb_data,
        !p_tracer_data->estimate_luminance
      );
      if (p_tracer_data->screen_shot) {
        std::vector<float> pixel_data(film->getWidth()* film->getHeight() * 3);
        CUdeviceptr ptr = (CUdeviceptr)owlBufferGetPointer(p_tracer_data->frame_buffer, 0);
        cuMemcpyDtoHAsync(pixel_data.data(), (CUdeviceptr)ptr, sizeof(float)* pixel_data.size(), owlContextGetStream(p_tracer_data->context, 0));

        std::vector<float> pixel_data_rev(film->getWidth()* film->getHeight() * 3);
        for (size_t i = 0; i < film->getHeight(); ++i) {
          std::memcpy(pixel_data_rev.data() + i * 3 * film->getWidth(), pixel_data.data() + (film->getHeight() - 1 - i) * 3 * film->getWidth(), 3 * film->getWidth() * sizeof(float));
        }

        auto desc = hikari::BitmapImageDesc();
        desc.depth_or_layers = 1;
        desc.width_in_bytes = film->getWidth() * sizeof(float) * 3;
        desc.height = film->getHeight();
        desc.x = 0; desc.y = 0; desc.z = 0;
        desc.p_data = pixel_data_rev.data();
        auto mipmap = hikari::Mipmap::create2D(hikari::BitmapDataType::eF32, 3, 1, film->getWidth(), film->getHeight(), { desc });
        hikari::ImageExporter::save(p_tracer_data->screen_filename, mipmap);
        p_tracer_data->screen_shot = false;
      }
      p_tracer_data->accum_sample++;
      };
    auto ui_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer) {
      TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
      if (ImGui::Begin("Config")) {
        if (ImGui::TreeNode("Frame")) {
          char temp[256] = {};
          std::memcpy(temp, p_tracer_data->screen_filename.c_str(), p_tracer_data->screen_filename.size());
          if (ImGui::InputText("filename", temp, sizeof(temp))) {
            p_tracer_data->screen_filename = std::string(temp);
          }
          if (ImGui::Checkbox("save", &p_tracer_data->screen_shot)) {
          }
          ImGui::TreePop();
        }
        if (ImGui::TreeNode("Tonemap")) {
          float new_key_value = p_tracer_data->p_tonemap->getKeyValue();
          float old_key_value = new_key_value;
          {
            const char* combo_defaults[] = { "Linear","Linear(Correlated)","Reinhard(Correlated","Extended Reinhard(Correlated)" };
            if (ImGui::BeginCombo("Type", combo_defaults[(int)p_tracer_data->p_tonemap->getType()])) {
              if (ImGui::Selectable("Linear")) {
                p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eLinear);
              }
              if (ImGui::Selectable("Linear(Correlated)")) {
                p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedLinear);
              }
              if (ImGui::Selectable("Reinhard(Correlated)")) {
                p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedReinhard);
              }
              if (ImGui::Selectable("Extended Reinhard(Correlated)")) {
                p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedExtendedReinhard);
              }
              ImGui::EndCombo();
            }
          }
          ImGui::SliderFloat("Key Value: ", &new_key_value, 0.001f, 5.0f);
          ImGui::Text("Maxmimum Luminance: %f", p_tracer_data->p_tonemap->getMaxLuminance());
          ImGui::Text("Average  Luminance: %f", p_tracer_data->p_tonemap->getAveLuminance());
          if (new_key_value != old_key_value) {
            p_tracer_data->p_tonemap->setKeyValue(new_key_value);
          }
          bool v = p_tracer_data->estimate_luminance;
          if (ImGui::Checkbox("Estimate Luminance", &v)) {
            p_tracer_data->estimate_luminance = v;
          }
          ImGui::TreePop();
        }
      }
      ImGui::End();
      };
    auto viewer = std::make_unique<hikari::test::owl::testlib::GLViewer>(owlContextGetStream(context, 0), film->getWidth(), film->getHeight());
    viewer->runWithCallback(&tracer_data, resize_callback, presskey_callback, press_mouse_button_callback, mouse_scroll_callback, update_callback, render_callback, ui_callback);
    viewer.reset();
    tonemap.free();
  }
  return 0;
}
