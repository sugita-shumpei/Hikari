#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/color.h>
#include <hikari/core/utils.h>
#include <hikari/core/scene.h>
#include <hikari/core/transform.h>
#include <hikari/core/node.h>
#include <hikari/core/bsdf.h>
#include <hikari/core/light.h>
#include <hikari/core/texture.h>
#include <hikari/core/spectrum.h>
#include <hikari/core/variant.h>
#include <tinyxml2.h>
#include <unordered_map>
#define HK_MITSUBA_XML_FOR_EACH(CHILD, PARENT) \
for (const tinyxml2::XMLElement* CHILD=PARENT->FirstChildElement();CHILD;CHILD=CHILD->NextSiblingElement())
#define HK_MITSUBA_XML_FOR_EACH_OF(CHILD, PARENT,NAME) \
for (const tinyxml2::XMLElement* CHILD=PARENT->FirstChildElement(NAME);CHILD;CHILD=CHILD->NextSiblingElement(NAME))

namespace hikari {
  // 各プラグインに含まれる可能性のある全ての変数を中間データ構造が保持
  // 中間データ構造を最終的な構造へ変換することで, XMLのロードを実現
  struct MitsubaXMLProperties;
  using  MitsubaXMLFloat   = F32;
  using  MitsubaXMLInteger = I32;
  using  MitsubaXMLString  = String;
  using  MitsubaXMLBoolean = Bool;
  struct MitsubaXMLRgb     { Vec3 color; };
  struct MitsubaXMLArray   { std::vector<F32> values; };
  using  MitsubaXMLPoint  = Vec3;
  using  MitsubaXMLVector = MitsubaXMLPoint;
  struct MitsubaXMLDefault {
    MitsubaXMLString name;
    MitsubaXMLString value;
  };
  struct MitsubaXMLVersion {
    U32              major;
    U32              minor;
    U32              patch;
  };
  struct MitsubaXMLRef {
    MitsubaXMLString id;
  };
  struct MitsubaXMLTranslate {
    Vec3 value;
  };
  struct MitsubaXMLRotate {
    Vec3 value;
    F32  angle;
  };
  struct MitsubaXMLScale {
    Vec3 value;
  };
  struct MitsubaXMLMatrix {
    std::vector<F32> values;
  };
  struct MitsubaXMLLookAt {
    Vec3 origin;
    Vec3 target;
    Vec3 up;
  };
  struct MitsubaXMLTransformElement {

    std::variant<MitsubaXMLTranslate, MitsubaXMLRotate, MitsubaXMLScale, MitsubaXMLMatrix, MitsubaXMLLookAt> data;
  };
  struct MitsubaXMLTransform {
    std::vector<MitsubaXMLTransformElement> elements;
  };
  struct MitsubaXMLTexture;
  using  MitsubaXMLTexturePtr = std::shared_ptr<MitsubaXMLTexture>;
  struct MitsubaXMLSpectrumInlineUniform {
    F32                             value;
  };
  struct MitsubaXMLSpectrumInlineWavelengthsAndValues {
    std::vector<F32> wavelengths;
    std::vector<F32> values;
  };
  struct MitsubaXMLSpectrumInlineFile {
    String                          filename;
  };
  struct MitsubaXMLSpectrumPlugin {
    String                       type;
    std::optional<F32>           wavelength_min;
    std::optional<F32>           wavelength_max;
    std::optional<F32>           temperature;
    std::optional<F32>           value;
    std::optional<F32>           scale;
    std::optional<MitsubaXMLRgb> color;
    std::optional<MitsubaXMLRef> ref;
    std::vector<F32>             wavelengths;
    std::vector<F32>             values;
    MitsubaXMLTexturePtr         texture;
  };
  struct MitsubaXMLSpectrum {
    std::variant<
      MitsubaXMLSpectrumInlineUniform,
      MitsubaXMLSpectrumInlineWavelengthsAndValues,
      MitsubaXMLSpectrumInlineFile,
      MitsubaXMLSpectrumPlugin
    > data;
  };
  struct MitsubaXMLProperties {
    std::unordered_map<MitsubaXMLString, MitsubaXMLInteger >   ingegers    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLFloat   >   floats      = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLString  >   strings     = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLPoint>      points      = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLVector>     vectors     = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLBoolean >   booleans    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLTexturePtr> textures    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLSpectrum>   spectrums   = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLRgb>        rgbs        = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLRef>        refs        = {};
    std::vector<MitsubaXMLRef>                                 nested_refs = {};
    std::vector<MitsubaXMLTexturePtr>                          nested_texs = {};
  };
  struct MitsubaXMLTexture {
    String               type;
    String               id;
    MitsubaXMLProperties properties;
  };
  struct MitsubaXMLIntegrator   {
    MitsubaXMLString     type;
    MitsubaXMLProperties properties;
  };
  struct MitsubaXMLFilm         {
    MitsubaXMLString     type;
    MitsubaXMLProperties properties;
  };
  struct MitsubaXMLSampler {
    MitsubaXMLString     type;
    MitsubaXMLProperties properties;
  };
  struct MitsubaXMLSensor       {
    MitsubaXMLString                   type;
    MitsubaXMLProperties               properties;
    std::optional<MitsubaXMLTransform> to_world;
    MitsubaXMLFilm                     film;
    MitsubaXMLSampler                  sampler;
  };
  struct MitsubaXMLEmitter      {
    MitsubaXMLString                   type;
    MitsubaXMLProperties               properties;
    std::optional<MitsubaXMLTransform> to_world;
  };
  using  MitsubaXMLEmitterPtr = std::shared_ptr<MitsubaXMLEmitter>;
  struct MitsubaXMLBsdf;
  using  MitsubaXMLBsdfPtr = std::shared_ptr<MitsubaXMLBsdf>;
  struct MitsubaXMLBsdf {
    MitsubaXMLString                  type;
    MitsubaXMLString                  id;
    MitsubaXMLProperties              properties;
    std::vector<MitsubaXMLBsdfPtr>    nested_bsdfs;
  };
  struct MitsubaXMLShape {
    MitsubaXMLString                   type;
    MitsubaXMLString                   id;
    MitsubaXMLProperties               properties;
    std::optional<MitsubaXMLTransform> to_world;
    MitsubaXMLBsdfPtr                  bsdf;
    MitsubaXMLEmitterPtr               emitter;
  };
  using  MitsubaXMLShapePtr = std::shared_ptr<MitsubaXMLShape>;
  // 中間データ構造(XMLのままだと変換が大変なため, ある程度扱いやすい単位にまとめておく)
  struct MitsubaXMLData   {
    String                                                    filepath;
    MitsubaXMLVersion                                         version     ;
    std::unordered_map<MitsubaXMLString,MitsubaXMLString>     defaults    ;
    MitsubaXMLIntegrator                                      integrator  ;
    MitsubaXMLSensor                                          sensor      ;// Sceneに紐づいたセンサ
    std::vector<MitsubaXMLShapePtr>                           shapes      ;// Sceneに紐づいた形状
    std::vector<MitsubaXMLEmitterPtr>                         emitters    ;// Sceneに紐づいた光源
    std::unordered_map<MitsubaXMLString,MitsubaXMLShapePtr>   ref_shapes  ;// 参照可能な形状
    std::unordered_map<MitsubaXMLString,MitsubaXMLBsdfPtr >   ref_bsdfs   ;// 参照可能なBSDF
    std::unordered_map<MitsubaXMLString,MitsubaXMLTexturePtr> ref_textures;// 参照可能なテクスチャ
  };
  struct MitsubaXMLParser : public tinyxml2::XMLVisitor {
    MitsubaXMLParser();
    virtual ~MitsubaXMLParser();
    virtual bool VisitEnter(const tinyxml2::XMLDocument& doc) override;
    virtual bool VisitExit(const tinyxml2::XMLDocument&  doc) override;
    auto getData() const -> const MitsubaXMLData&;

    void setFilepath(const String& filepath);
    bool parseVersion(const tinyxml2::XMLElement*    element_scene);
    bool parseDefault(const tinyxml2::XMLElement*    element_scene);
    bool parseInteger(const tinyxml2::XMLElement*    element_int, MitsubaXMLString& name, MitsubaXMLInteger& value);
    bool parseFloat  (const tinyxml2::XMLElement*    element_flt, MitsubaXMLString& name, MitsubaXMLFloat  & value);
    bool parsePointOrVector(const tinyxml2::XMLElement*    element_pnt, MitsubaXMLString& name, MitsubaXMLPoint  & value);
    bool parseBoolean   (const tinyxml2::XMLElement* element_b8 , MitsubaXMLString& name, MitsubaXMLBoolean& value);
    bool parseString    (const tinyxml2::XMLElement* element_str, MitsubaXMLString& name, MitsubaXMLString & value);
    bool parseSpectrumPlugin(const tinyxml2::XMLElement* element_spe, MitsubaXMLSpectrumPlugin& value);
    bool parseSpectrum  (const tinyxml2::XMLElement* element_spe, MitsubaXMLString& name, MitsubaXMLSpectrum&value);
    bool parseRgb       (const tinyxml2::XMLElement* element_rgb, MitsubaXMLString& name, MitsubaXMLRgb    & value);
    bool parseRef       (const tinyxml2::XMLElement* element_ref, MitsubaXMLString   & name, MitsubaXMLRef    & value);
    bool parseTranslate (const tinyxml2::XMLElement* element_tra, MitsubaXMLTranslate& translate);
    bool parseRotate    (const tinyxml2::XMLElement* element_rot, MitsubaXMLRotate& rotate);
    bool parseScale     (const tinyxml2::XMLElement* element_scl, MitsubaXMLScale & scale);
    bool parseMatrix    (const tinyxml2::XMLElement* element_mat, MitsubaXMLMatrix& matrix);
    bool parseLookAt    (const tinyxml2::XMLElement* element_lka, MitsubaXMLLookAt& lookat);
    bool parseTransformElement(const tinyxml2::XMLElement* element_trf, MitsubaXMLTransformElement& element);
    bool parseTransform (const tinyxml2::XMLElement* element_trf, MitsubaXMLString& name, MitsubaXMLTransform& transform);
    bool parseProperties(const tinyxml2::XMLElement* element_object, MitsubaXMLProperties& properties);
    bool parseObject    (const tinyxml2::XMLElement* element_object, MitsubaXMLString& type, MitsubaXMLString& name, MitsubaXMLString& id, MitsubaXMLProperties& properties);
    bool parseIntegrator(const tinyxml2::XMLElement* element_scene);
    bool parseSensor    (const tinyxml2::XMLElement* element_scene);
    bool parseShape     (const tinyxml2::XMLElement* element_shape, std::shared_ptr<MitsubaXMLShape>& shape);
    bool parseShapes    (const tinyxml2::XMLElement* element_scene);
    bool parseBsdf      (const tinyxml2::XMLElement* element_bsdf , std::shared_ptr<MitsubaXMLBsdf>& bsdf);
    bool parseBsdfs     (const tinyxml2::XMLElement* element_scene);
    bool parseTexture(const tinyxml2::XMLElement* element_texture, MitsubaXMLString& name, std::shared_ptr<MitsubaXMLTexture>& texture);
    bool parseTextures(const tinyxml2::XMLElement* element_scene);
    bool parseEmitter   (const tinyxml2::XMLElement* element_emitter, std::shared_ptr<MitsubaXMLEmitter>& emitter);
    bool parseEmitters  (const tinyxml2::XMLElement* element_scene);
    auto normalizeString(const String& input) const -> String;
    void solveTexture(std::shared_ptr<MitsubaXMLTexture>& texture);
    void solveShape(std::shared_ptr<MitsubaXMLShape>& shape);
    void solveBsdf(std::shared_ptr<MitsubaXMLBsdf>& bsdf);
  private:
    MitsubaXMLData        m_data;
  };
}
