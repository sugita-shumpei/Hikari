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
  struct MitsubaXMLProperties {
    std::unordered_map<MitsubaXMLString, MitsubaXMLInteger >   ingegers    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLFloat   >   floats      = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLString  >   strings     = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLBoolean >   booleans    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLTexturePtr> textures    = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLRgb>        rgbs        = {};
    std::unordered_map<MitsubaXMLString, MitsubaXMLRef>        refs        = {};
    std::vector<MitsubaXMLRef>                                 nested_refs = {};
  };
  struct MitsubaXMLTexture {
    String               type;
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
    MitsubaXMLString               type;
    MitsubaXMLProperties           properties;
    std::vector<MitsubaXMLBsdfPtr> nested_bsdfs;
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
  struct MitsubaXMLData   {
    MitsubaXMLString                                          filepath  ;
    MitsubaXMLVersion                                         version   ;
    std::unordered_map<MitsubaXMLString,MitsubaXMLString>     defaults  ;
    MitsubaXMLIntegrator                                      integrator;
    MitsubaXMLSensor                                          sensor    ;
    std::vector<MitsubaXMLShapePtr>                           shapes    ;
    std::vector<MitsubaXMLEmitterPtr>                         emitters  ;
    std::unordered_map<MitsubaXMLString,MitsubaXMLShapePtr>   ref_shapes;
    std::unordered_map<MitsubaXMLString,MitsubaXMLBsdfPtr >   ref_bsdfs ;
    std::unordered_map<MitsubaXMLString,MitsubaXMLTexturePtr> ref_textures;
  };
  
  struct MitsubaXMLParser : public tinyxml2::XMLVisitor {
    virtual ~MitsubaXMLParser() {}
    virtual bool VisitEnter(const tinyxml2::XMLDocument& doc) override {
      auto element_scene = doc.FirstChildElement("scene");
      if (!parseVersion(element_scene)) { return false; }// Versionを取得
      if (!parseDefault(element_scene)) { return false; }// Defaultを取得
      if (!parseIntegrator(element_scene)) { return false; }
      if (!parseSensor(element_scene)) { return false; }
      if (!parseTextures(element_scene)) { return false; }
      if (!parseBsdfs (element_scene)) { return false; }
      if (!parseEmitters(element_scene)) { return false; }
      if (!parseShapes(element_scene)) { return false; }
      return true;
    }
    virtual bool VisitExit(const tinyxml2::XMLDocument&  doc) override {
      for (auto& shape : m_data.shapes) {
        solveShape(shape);
      }
      return true;
    }
    auto getData() const -> const MitsubaXMLData& { return m_data; }

    bool parseVersion(const tinyxml2::XMLElement*    element_scene) {
      if (!element_scene) { return false; }
      auto attr_version = element_scene->FindAttribute("version");
      if (!attr_version) { return false; }
      auto version_strs = hikari::splitString(attr_version->Value(), '.');
      if (version_strs.size() != 3) { return false; }
      MitsubaXMLVersion tmp_version = {};
      try {
        tmp_version.major = std::stoi(version_strs[0]);
        tmp_version.minor = std::stoi(version_strs[1]);
        tmp_version.patch = std::stoi(version_strs[2]);
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      m_data.version = tmp_version;
      return true;
    }
    bool parseDefault(const tinyxml2::XMLElement*    element_scene) {
      std::unordered_map<MitsubaXMLString, MitsubaXMLString> tmp_defaults;
      HK_MITSUBA_XML_FOR_EACH_OF(element_default, element_scene, "default") {
        if (!element_default) { return false; }
        auto attr_name  = element_default->FindAttribute("name");
        auto attr_value = element_default->FindAttribute("value");
        if (!attr_name || !attr_value) { return false; }
        tmp_defaults.insert({ attr_name->Value(),attr_value->Value() });
      }
      m_data.defaults = tmp_defaults;
      return true;
    }
    bool parseInteger(const tinyxml2::XMLElement*    element_int, MitsubaXMLString& name, MitsubaXMLInteger& value) {
      if (!element_int) { return false; }
      auto attr_name  = element_int->FindAttribute("name");
      auto attr_value = element_int->FindAttribute("value");
      if (!attr_name || !attr_value) { return false; }
      auto name_str   = normalizeString(attr_name->Value());
      auto value_str  = normalizeString(attr_value->Value());
      MitsubaXMLInteger tmp_value = {};
      try {
        tmp_value = std::stoi(value_str);
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      name  = name_str;
      value = tmp_value;
      return true;
    }
    bool parseFloat  (const tinyxml2::XMLElement*    element_int, MitsubaXMLString& name, MitsubaXMLFloat  & value) {
      if (!element_int) { return false; }
      auto attr_name = element_int->FindAttribute("name");
      auto attr_value = element_int->FindAttribute("value");
      if (!attr_name || !attr_value) { return false; }
      auto name_str = normalizeString(attr_name->Value());
      auto value_str = normalizeString(attr_value->Value());
      MitsubaXMLFloat tmp_value = {};
      try {
        tmp_value = std::stof(value_str);
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      name = name_str;
      value = tmp_value;
      return true;
    }
    bool parseBoolean   (const tinyxml2::XMLElement* element_int, MitsubaXMLString& name, MitsubaXMLBoolean& value) {
      if (!element_int) { return false; }
      auto attr_name = element_int->FindAttribute("name");
      auto attr_value = element_int->FindAttribute("value");
      if (!attr_name || !attr_value) { return false; }
      auto name_str   = normalizeString(attr_name->Value());
      auto value_str = normalizeString(attr_value->Value());
      MitsubaXMLBoolean tmp_value = {};
      do {
        if (value_str == "true" ) { tmp_value = true; break; }
        if (value_str == "false") { tmp_value = false; break; }
        return false;
      } while (0);
      name  = name_str;
      value = tmp_value;
      return true;
    }
    bool parseString    (const tinyxml2::XMLElement* element_int, MitsubaXMLString& name, MitsubaXMLString & value) {
      if (!element_int) { return false; }
      auto attr_name = element_int->FindAttribute("name");
      auto attr_value = element_int->FindAttribute("value");
      if (!attr_name || !attr_value) { return false; }
      name  = normalizeString(attr_name->Value());
      value = normalizeString(attr_value->Value());
      return true;
    }
    bool parseRgb       (const tinyxml2::XMLElement* element_rgb, MitsubaXMLString& name, MitsubaXMLRgb    & value) {
      if (!element_rgb) { return false; }
      auto attr_name  = element_rgb->FindAttribute("name");
      auto attr_value = element_rgb->FindAttribute("value");
      if (!attr_value||!attr_name) { return false; }
      auto value_strs = splitString(normalizeString(attr_value->Value()),',');
      if (value_strs.size() != 1 && value_strs.size() != 3) { return false; }
      auto values = std::vector<F32>();
      values.reserve(value_strs.size());
      try {
        for (size_t i = 0; i < value_strs.size(); ++i) {
          values.push_back(std::stof(value_strs[i]));
        }
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      if (values.size() == 1) { value.color = Vec3(values[0]); }
      if (values.size() == 3) { value.color = Vec3(values[0], values[1], values[2]); }
      name = normalizeString(attr_name->Value());
      return true;
    }
    bool parseRef       (const tinyxml2::XMLElement* element_ref, MitsubaXMLString   & name, MitsubaXMLRef    & value){
      if (!element_ref) { return false; }
      auto attr_name  = element_ref->FindAttribute("name");
      auto attr_id    = element_ref->FindAttribute("id");
      if (!attr_id) { return false; }
      value.id = normalizeString(attr_id->Value());
      if (attr_name) {
        name = normalizeString(attr_name->Value());
      }
      else {
        name = "";
      }
      return true;
    }
    bool parseTranslate (const tinyxml2::XMLElement* element_tra, MitsubaXMLTranslate& translate) {
      if (!element_tra) { return false; }
      auto attr_value = element_tra->FindAttribute("value");
      if (attr_value ) {
        auto value_strs = splitString(normalizeString(attr_value->Value()), ',');
        if (value_strs.size() != 1 && value_strs.size() != 3) { return false; }
        auto values = std::vector<F32>();
        values.reserve(value_strs.size());
        try {
          for (size_t i = 0; i < value_strs.size(); ++i) {
            values.push_back(std::stof(value_strs[i]));
          }
        }
        catch (std::invalid_argument& err) { return false; }
        catch (std::out_of_range& err) { return false; }
        if (values.size() == 1) { translate.value = Vec3(values[0]); }
        if (values.size() == 3) { translate.value = Vec3(values[0], values[1], values[2]); }
      }
      else {
        translate.value = {};
      }
      auto attr_x = element_tra->FindAttribute("x");
      auto attr_y = element_tra->FindAttribute("y");
      auto attr_z = element_tra->FindAttribute("z");
      try {
        if (attr_x) {
          auto value_x = std::stof(normalizeString(attr_x->Value()));
          translate.value.x = value_x;
        }
        if (attr_y) {
          auto value_y = std::stof(normalizeString(attr_y->Value()));
          translate.value.y = value_y;
        }
        if (attr_z) {
          auto value_z = std::stof(normalizeString(attr_z->Value()));
          translate.value.z = value_z;
        }
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      return true;
    }
    bool parseRotate    (const tinyxml2::XMLElement* element_rot, MitsubaXMLRotate& rotate) {
      if (!element_rot) { return false; }
      auto attr_angle = element_rot->FindAttribute("angle");
      if (!attr_angle) { return false; }
      try {
        rotate.angle = std::stof(normalizeString(attr_angle->Value()));
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      auto attr_value = element_rot->FindAttribute("value");
      if ( attr_value) {
        auto value_strs = splitString(normalizeString(attr_value->Value()), ',');
        if (value_strs.size() != 1 && value_strs.size() != 3) { return false; }
        auto values = std::vector<F32>();
        values.reserve(value_strs.size());
        try {
          for (size_t i = 0; i < value_strs.size(); ++i) {
            values.push_back(std::stof(value_strs[i]));
          }
        }
        catch (std::invalid_argument& err) { return false; }
        catch (std::out_of_range& err) { return false; }
        if (values.size() == 1) { rotate.value = Vec3(values[0]); }
        if (values.size() == 3) { rotate.value = Vec3(values[0], values[1], values[2]); }
      }
      else {
        rotate.value = { 0.0f,0.0f,0.0f };
      }
      auto attr_x = element_rot->FindAttribute("x");
      auto attr_y = element_rot->FindAttribute("y");
      auto attr_z = element_rot->FindAttribute("z");
      try {
        if (attr_x){
        auto value_x   = std::stof(normalizeString(attr_x->Value()));
        rotate.value.x = value_x;
      }
        if (attr_y) {
        auto value_y   = std::stof(normalizeString(attr_y->Value()));
        rotate.value.y = value_y;
      }
        if (attr_z) {
        auto value_z   = std::stof(normalizeString(attr_z->Value()));
        rotate.value.z = value_z;
      }
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      return true;
    }
    bool parseScale     (const tinyxml2::XMLElement* element_scl, MitsubaXMLScale & scale) {
      if (!element_scl) { return false; }
      auto attr_value = element_scl->FindAttribute("value");
      if (!attr_value) {
        return false;
      }
      auto value_strs = splitString(normalizeString(attr_value->Value()), ',');
      if (value_strs.size() != 1 && value_strs.size() != 3) { return false; }
      auto values = std::vector<F32>();
      values.reserve(value_strs.size());
      try {
        for (size_t i = 0; i < value_strs.size(); ++i) {
          values.push_back(std::stof(value_strs[i]));
        }
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      if (values.size() == 1) { scale.value = Vec3(values[0]); }
      if (values.size() == 3) { scale.value = Vec3(values[0], values[1], values[2]); }
      return true;
    }
    bool parseMatrix    (const tinyxml2::XMLElement* element_mat, MitsubaXMLMatrix& matrix) {
      if (!element_mat) { return false; }
      auto attr_value = element_mat->FindAttribute("value");
      if (!attr_value) { return false; }
      auto value_strs = splitString(normalizeString(attr_value->Value()), ' ');
      if (value_strs.size() != 16 && value_strs.size() != 9) { return false; }
      auto values = std::vector<F32>();
      values.reserve(value_strs.size());
      try {
        for (size_t i = 0; i < value_strs.size(); ++i) {
          values.push_back(std::stof(value_strs[i]));
        }
      }
      catch (std::invalid_argument& err) { return false; }
      catch (std::out_of_range& err) { return false; }
      matrix.values = values;
      return true;
    }
    bool parseLookAt    (const tinyxml2::XMLElement* element_lka, MitsubaXMLLookAt& lookat) {
      if (!element_lka) { return false; }
      auto attr_origin = element_lka->FindAttribute("origin");
      auto attr_target = element_lka->FindAttribute("target");
      auto attr_up     = element_lka->FindAttribute("up");
      if (!attr_origin || !attr_target) { return false; }
      auto origin_strs = splitString(normalizeString(attr_origin->Value()), ',');
      auto target_strs = splitString(normalizeString(attr_target->Value()), ',');
      if (origin_strs.size() != 1 && origin_strs.size() != 3) { return false; }
      if (target_strs.size() != 1 && target_strs.size() != 3) { return false; }
      {
        auto values = std::vector<F32>();
        values.reserve(origin_strs.size());
        try {
          for (size_t i = 0; i < origin_strs.size(); ++i) {
            values.push_back(std::stof(origin_strs[i]));
          }
        }
        catch (std::invalid_argument& err) { return false; }
        catch (std::out_of_range& err) { return false; }
        if (values.size() == 1) { lookat.origin = Vec3(values[0]); }
        if (values.size() == 3) { lookat.origin = Vec3(values[0], values[1], values[2]); }
      }
      {
        auto values = std::vector<F32>();
        values.reserve(target_strs.size());
        try {
          for (size_t i = 0; i < origin_strs.size(); ++i) {
            values.push_back(std::stof(target_strs[i]));
          }
        }
        catch (std::invalid_argument& err) { return false; }
        catch (std::out_of_range& err) { return false; }
        if (values.size() == 1) { lookat.target = Vec3(values[0]); }
        if (values.size() == 3) { lookat.target = Vec3(values[0], values[1], values[2]); }
      }
      if (attr_up) {
        auto up_strs = splitString(normalizeString(attr_up->Value()), ',');

        if (up_strs.size() != 1 && up_strs.size() != 3) { return false; }
        auto values = std::vector<F32>();
        values.reserve(up_strs.size());
        try {
          for (size_t i = 0; i < up_strs.size(); ++i) {
            values.push_back(std::stof(up_strs[i]));
          }
        }
        catch (std::invalid_argument& err) { return false; }
        catch (std::out_of_range& err) { return false; }
        if (values.size() == 1) { lookat.up = Vec3(values[0]); }
        if (values.size() == 3) { lookat.up = Vec3(values[0], values[1], values[2]); }
      }
      else {
        lookat.up = { 0.0f,1.0f,0.0f };
      }
      return true;
    }
    bool parseTransformElement(const tinyxml2::XMLElement* element_trf, MitsubaXMLTransformElement& element) {
      if (!element_trf) { return false; }
      if (element_trf->Name() == std::string_view("translate")) {
        MitsubaXMLTranslate trs;
        if (parseTranslate(element_trf, trs)) { element.data = trs; }
        return true;
      }
      if (element_trf->Name() == std::string_view("rotate")) {
        MitsubaXMLRotate rot;
        if (parseRotate(element_trf, rot)) { element.data = rot; }
        return true;
      }
      if (element_trf->Name() == std::string_view("scale")) {
        MitsubaXMLScale scl;
        if (parseScale(element_trf, scl)) { element.data = scl; }
        return true;
      }
      if (element_trf->Name() == std::string_view("matrix")) {
        MitsubaXMLMatrix mat;
        if (parseMatrix(element_trf, mat)) { element.data = mat; }
        return true;
      }
      if (element_trf->Name() == std::string_view("lookat")) {
        MitsubaXMLLookAt lka;
        if (parseLookAt(element_trf, lka)) { element.data = lka; }
        return true;
      }
      return true;
    }
    bool parseTransform(const tinyxml2::XMLElement* element_trf, MitsubaXMLString& name, MitsubaXMLTransform& transform) {
      if (!element_trf) { return false; }
      auto attr_name = element_trf->FindAttribute("name");
      if (attr_name){ name = normalizeString(attr_name->Value()); }
      if (element_trf->Name() != std::string_view("transform")) { return false; }
      HK_MITSUBA_XML_FOR_EACH(element_child, element_trf) {
        MitsubaXMLTransformElement elem;
        elem.data = MitsubaXMLMatrix();
        if (!parseTransformElement(element_child, elem)) {}
        transform.elements.push_back(elem);
      }
      return true;
    }
    bool parseProperties(const tinyxml2::XMLElement* element_object, MitsubaXMLProperties& properties) {
      HK_MITSUBA_XML_FOR_EACH_OF(element_int , element_object, "integer") {
        String name; I32 value;
        if (!parseInteger(element_int, name, value)) { return false; }
        properties.ingegers.insert({ name,value });
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_flt , element_object, "float"  ) {
        String name; F32 value;
        if (!parseFloat(element_flt, name, value)) { return false; }
        properties.floats.insert({ name,value });
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_str , element_object, "string" ) {
        String name; String value;
        if (!parseString(element_str, name, value)) { return false; }
        properties.strings.insert({ name,value });
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_bool, element_object, "boolean") {
        String name; Bool value;
        if (!parseBoolean(element_bool, name, value)) { return false; }
        properties.booleans.insert({ name,value });
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_ref , element_object, "ref"    ) {
        String name; MitsubaXMLRef value;
        if (!parseRef(element_ref, name, value)) { return false; }
        if (name != "") { properties.refs.insert({ name,value }); }
        else { properties.nested_refs.push_back(value); }
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_rgb, element_object , "rgb"    ) {
        String name; MitsubaXMLRgb value;
        if (!parseRgb(element_rgb, name, value)) { return false; }
        properties.rgbs.insert({ name,value });
      }
      HK_MITSUBA_XML_FOR_EACH_OF(element_tex, element_object, "texture") {
        String name; auto tmp_texture = std::make_shared<MitsubaXMLTexture>();
        if (!parseTexture(element_tex, name, tmp_texture)) { return false; }
        properties.textures.insert({ name,tmp_texture });
      }
      return true;
    }
    bool parseObject    (const tinyxml2::XMLElement* element_object, MitsubaXMLString& type, MitsubaXMLString& name, MitsubaXMLString& id, MitsubaXMLProperties& properties)
    {
      type       = "";
      name       = "";
      id         = "";
      properties = {};
      auto attr_type = element_object->FindAttribute("type");
      if (!element_object) { return false; }
      if (!attr_type) { return false; }
      type = normalizeString(attr_type->Value());
      if (!parseProperties(element_object, properties)) { return false; }
      auto attr_name = element_object->FindAttribute("name");
      auto attr_id   = element_object->FindAttribute("id");
      if (attr_name) { name = normalizeString(attr_name->Value()); }
      if (attr_id)   { id = normalizeString(attr_id->Value()); }
      return true;
    }
    bool parseIntegrator(const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      auto element_integrator = element_scene->FirstChildElement("integrator");
      std::string name;
      std::string id;
      return parseObject(element_integrator,m_data.integrator.type,name,id,m_data.integrator.properties);
    }
    bool parseSensor    (const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      auto element_sensor = element_scene->FirstChildElement("sensor");
      std::string name;
      std::string id;
      if (!parseObject(element_sensor, m_data.sensor.type, name, id, m_data.sensor.properties)) { return false; }
      {
        auto element_tran = element_sensor->FirstChildElement("transform");
        if (element_tran) {
          std::string name;
          MitsubaXMLTransform transform;
          if (parseTransform(element_tran, name, transform)) {
            m_data.sensor.to_world = transform;
          }
        }
      }
      auto element_film    = element_sensor->FirstChildElement("film");
      if (!parseObject(element_film  , m_data.sensor.film.type, name, id, m_data.sensor.film.properties)) { return false; }
      auto element_sampler = element_sensor->FirstChildElement("sampler");
      if (!parseObject(element_sampler, m_data.sensor.sampler.type, name, id, m_data.sensor.sampler.properties)) { return false; }
      return true;
    }
    bool parseShape     (const tinyxml2::XMLElement* element_shape, std::shared_ptr<MitsubaXMLShape>& shape){
      if (!shape) { return false; }
      std::string name;
      std::string id;
      if (!parseObject(element_shape, shape->type, name, id, shape->properties)) { return false; }
      {
        auto element_tran = element_shape->FirstChildElement("transform");
        if  (element_tran) {
          std::string name;
          MitsubaXMLTransform transform;
          if (parseTransform(element_tran,name,transform)) {
            shape->to_world = transform;
          }
        }
      }
      if (id != "") { shape->id = id; } else { shape->id = ""; }
      if (id != "") { m_data.ref_shapes.insert({ id,shape }); }
      {
        auto tmp_bsdf = std::make_shared<MitsubaXMLBsdf>();
        if (parseBsdf(element_shape->FirstChildElement("bsdf"), tmp_bsdf)) {
          shape->bsdf = tmp_bsdf;
        }
      }
      {
        auto tmp_emitter = std::make_shared<MitsubaXMLEmitter>();
        if (parseEmitter(element_shape->FirstChildElement("emitter"), tmp_emitter)) {
          shape->emitter = tmp_emitter;
        }
      }
      return true;
    }
    bool parseShapes    (const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      HK_MITSUBA_XML_FOR_EACH_OF(element_shape, element_scene, "shape") {
        auto tmp_shape = std::make_shared<MitsubaXMLShape>();
        if (!parseShape(element_shape, tmp_shape)) { continue; }
        m_data.shapes.push_back(tmp_shape);
      }
      return true;
    }
    bool parseBsdf      (const tinyxml2::XMLElement* element_bsdf , std::shared_ptr<MitsubaXMLBsdf>& bsdf){
      if (!element_bsdf || !bsdf) { return false; }
      std::string name;
      std::string id;
      if (!parseObject(element_bsdf, bsdf->type, name, id, bsdf->properties)) { return false; }
      if (id != "") { m_data.ref_bsdfs.insert({ id,bsdf }); }
      {
        HK_MITSUBA_XML_FOR_EACH_OF(element_nested_bsdf, element_bsdf, "bsdf") {
          auto tmp_nested = std::make_shared<MitsubaXMLBsdf>();
          if (parseBsdf(element_nested_bsdf, tmp_nested)) {
            bsdf->nested_bsdfs.push_back(tmp_nested);
          }
        }
      }
      return true;
    }
    bool parseBsdfs     (const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      HK_MITSUBA_XML_FOR_EACH_OF(element_bsdf, element_scene, "bsdf") {
        auto tmp_bsdf = std::make_shared<MitsubaXMLBsdf>();
        if (!parseBsdf(element_bsdf, tmp_bsdf)) { continue; }
      }
      return true;
    }
    bool parseTexture(const tinyxml2::XMLElement* element_texture, MitsubaXMLString& name, std::shared_ptr<MitsubaXMLTexture>& texture) {
      if (!element_texture || !texture) { return false; }
      std::string id;
      if (!parseObject(element_texture, texture->type, name, id, texture->properties)) { return false; }
      if (id != "") { m_data.ref_textures.insert({ id,texture }); }
      return true;
    }
    bool parseTextures(const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      HK_MITSUBA_XML_FOR_EACH_OF(element_texture, element_scene, "texture") {
        auto tmp_texture = std::make_shared<MitsubaXMLTexture>();
        std::string name = "";
        if (!parseTexture(element_texture,name, tmp_texture)) { continue; }
      }
      return true;
    }
    bool parseEmitter   (const tinyxml2::XMLElement* element_emitter, std::shared_ptr<MitsubaXMLEmitter>& emitter) {
      if (!element_emitter||!emitter) { return false; }
      std::string name;
      std::string id;
      if (!parseObject(element_emitter, emitter->type, name, id, emitter->properties)) { return false; }
      {
        auto element_tran = element_emitter->FirstChildElement("transform");
        if (element_tran) {
          std::string name;
          MitsubaXMLTransform transform;
          if (parseTransform(element_tran, name, transform)) {
            emitter->to_world = transform;
          }
        }
      }
      return true;
    }
    bool parseEmitters  (const tinyxml2::XMLElement* element_scene) {
      if (!element_scene) { return false; }
      HK_MITSUBA_XML_FOR_EACH_OF(element_emitter, element_scene, "emitter") {
        auto tmp_emitter = std::make_shared<MitsubaXMLEmitter>();
        if (!parseEmitter(element_emitter, tmp_emitter)) { continue; }
        m_data.emitters.push_back(tmp_emitter);
      }
      return true;
    }
    void solveShape     (std::shared_ptr<MitsubaXMLShape>&   shape){
      if (shape->properties.nested_refs.size() > 0) {
        // bsdfが見つからないとき、Refを通じてBsdfの割り当てを試みる
        if (!shape->bsdf) {
          for (auto& nested_ref : shape->properties.nested_refs) {
            auto iter = m_data.ref_bsdfs.find(nested_ref.id);
            if (iter == m_data.ref_bsdfs.end()) { continue; }
            shape->bsdf = iter->second;
            break;
          }
        }
      }
    }
    auto normalizeString(const String& input) const -> String {
      auto pos_dollers = std::vector<size_t>();
      {
        auto pos = size_t(0);
        auto off = size_t(0);
        while (off < input.size()) {
          auto pos = input.find_first_of('$', off);
          if (pos == std::string::npos) { break; }
          pos_dollers.push_back(pos);
          off = pos + 1;
        }
      }
      if (pos_dollers.empty()) { return input; }
      auto replace_strs = std::vector<std::pair<size_t,String>> ();
      replace_strs.reserve(pos_dollers.size());
      for (auto& pos_doller : pos_dollers)
      {
        bool is_match = false;
        for (auto& [name, value] : m_data.defaults)
        {
          if (input.compare(pos_doller + 1, name.size(), name) != 0) {
            continue;
          }
          replace_strs.push_back({ name.size(),value });
          is_match = true;
        }
        if (!is_match) { return input; }
      }
      //置換を始める
      std::string res = "";
      std::size_t off = 0;
      for (size_t i = 0; i < pos_dollers.size(); ++i) {
        res += input.substr(off, pos_dollers[i] - off);
        auto&  inserter = replace_strs[i].second;
        res += inserter;
        off  = pos_dollers[i] + (1 + replace_strs[i].first);
      }
      if (off < input.size()) {
        res += input.substr(off, input.size() - off);
      }
      return res;
    }
  private:
    MitsubaXMLData        m_data;
  };
}
