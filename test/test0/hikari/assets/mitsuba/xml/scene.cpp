#include "scene.h"
#include "transform.h"
#include "spectrum.h"
#include "rfilter.h"
#include "volume.h"
#include <filesystem>
#include <sstream>
#include <scn/scn.h>
#include <fmt/printf.h>
#include <tinyxml2.h>

#define HK_MITSUBA_XML_FOR_EACH(CHILD, PARENT) \
for (const tinyxml2::XMLElement* CHILD=PARENT->FirstChildElement();CHILD;CHILD=CHILD->NextSiblingElement())
#define HK_MITSUBA_XML_FOR_EACH_OF(CHILD, PARENT,NAME) \
for (const tinyxml2::XMLElement* CHILD=PARENT->FirstChildElement(NAME);CHILD;CHILD=CHILD->NextSiblingElement(NAME))


auto hikari::assets::mitsuba::XMLScene::create() noexcept -> std::shared_ptr<XMLScene>
{
  return std::shared_ptr<XMLScene>(new XMLScene());
}

hikari::assets::mitsuba::XMLScene::~XMLScene() noexcept
{
}

auto hikari::assets::mitsuba::XMLScene::toString() const -> std::string
{
  using namespace std::string_literals;
  auto object_to_str = [this](XMLObjectPtr object) {
    if (!object){ return std::vector<std::string>{};}
    std::stringstream ss;
    if (!object) { return std::vector<std::string>{}; }
    ss << object->toString();
    std::string tmp;
    std::vector<std::string> res;
    while (std::getline(ss, tmp, '\n')) {
      res.push_back("  "s + tmp + "\n"s);
    }
    return res;
  };
  auto interator_strs = object_to_str(getIntegrator());
  auto sensor_strs    = object_to_str(getSensor());

  auto res = std::string("");
  res += "{\n";
  res += "  \"version\":\""   + m_own_context->getVersionString() + "\",\n";
  res += "  \"filepath\":\"" + m_own_context->getPath() + "\",\n";
  res += "  \"subpaths\":[\n";
  {
  auto idx = 0;
  auto subpaths = m_own_context->getSubPathes();
  for (auto& subpath : subpaths) {
    res += "    " + subpath;
    if (idx == subpaths.size() - 1) { res += "\n"; }else{ res += ",\n" ;} idx++;
  }
  }
  res += "  ],\n";
  res += "  \"ref\":{\n";
  {
    {
  res += "    \"shapes\":{\n";
      auto shapes = m_own_context->getRefShapes();
      auto idx = 0;
      for (auto& shape : shapes) {
        {
  res += "      \""s + shape->getID() + "\":\n";
          std::stringstream ss;
          ss << shape->toString(); std::string sentence;
          while(std::getline(ss,sentence,'\n')){
  res += "        "s + sentence;
             if (ss.rdbuf()->in_avail()){ res += "\n"; }
          }
          if (idx != shapes.size() - 1) { res += "      ,\n"; }else { res += "      \n"; }
          ++idx;
        }
      }
  res += "    },\n";
    }
    {
  res += "    \"bsdfs\":{\n";
      auto bsdfs = m_own_context->getRefBsdfs();
      auto idx = 0;
      for (auto& bsdf : bsdfs) {
        {
  res += "      \""s + bsdf->getID() + "\":\n";
          std::stringstream ss;
          ss << bsdf->toString(); std::string sentence;
          while (std::getline(ss, sentence, '\n')) {
  res += "        "s + sentence;
            if (ss.rdbuf()->in_avail()) { res += "\n"; }
          }
          if (idx != bsdfs.size() - 1) { res += ",\n"; }
          else { res += "\n"; }
          ++idx;
        }
      }
  res += "    },\n";
    }
    {
      res += "    \"mediums\":{\n";
      auto mediums = m_own_context->getRefMediums();
      auto idx = 0;
      for (auto& medium : mediums) {
        {
          res += "      \""s + medium->getID() + "\":\n";
          std::stringstream ss;
          ss << medium->toString(); std::string sentence;
          while (std::getline(ss, sentence, '\n')) {
            res += "        "s + sentence;
            if (ss.rdbuf()->in_avail()) { res += "\n"; }
          }
          if (idx != mediums.size() - 1) { res += ",\n"; }
          else { res += "\n"; }
          ++idx;
        }
      }
      res += "    },\n";
    }
    {
      res += "    \"textures\":{\n";
      auto textures = m_own_context->getRefTextures();
      auto idx = 0;
      for (auto& texture : textures) {
        {
          res += "      \""s + texture->getID() + "\":\n";
          std::stringstream ss;
          ss << texture->toString(); std::string sentence;
          while (std::getline(ss, sentence, '\n')) {
            res += "        "s + sentence;
            if (ss.rdbuf()->in_avail()) { res += "\n"; }
          }
          if (idx != textures.size() - 1) { res += ",\n"; }
          else { res += "\n"; }
          ++idx;
        }
      }
      res += "    }\n";
    }
  }
  res += "  },\n";
  res += "  \"interator\": \n";
  for (auto& str : interator_strs) {
    std::string sentence;
    std::stringstream ss;
    ss << str;
    while (std::getline(ss, sentence, '\n')) {
      res += "  " + str ;
      if (ss.rdbuf()->in_avail()) { res += "\n"; }
    }
  }
  res += "  ,\n";
  res += "  \"sensor\": \n";
  for (auto& str : sensor_strs) {
    std::string sentence;
    std::stringstream ss;
    ss << str;
    while (std::getline(ss, sentence, '\n')) {
      res += "  " + str;
      if (ss.rdbuf()->in_avail()) { res += "\n"; }
    }
  }
  res += "  ,\n";
  res += "  \"shapes\": \n";
  res += "  [\n";
  {
  auto idx = 0;
  for (auto& object : m_shapes) {
    auto object_strs = object_to_str(object);
    for (auto& str : object_strs) {
      std::string sentence;
      std::stringstream ss;
      ss << str;
      while (std::getline(ss, sentence, '\n')) {
        res += "  " + str;
        if (ss.rdbuf()->in_avail()) { res += "\n"; }
      }
    }
    if (idx == m_shapes.size()-1){
      res += "    \n";
    }
    else {
      res += "    ,\n";
    }
    idx++;
  }
  }
  res += "  ],\n";
  res += "  \"emitters\": \n";
  res += "  [\n";
  {
    auto idx = 0;
    for (auto& object : m_emitters) {
      auto object_strs = object_to_str(object);
      for (auto& str : object_strs) {
        std::string sentence;
        std::stringstream ss;
        ss << str;
        while (std::getline(ss, sentence, '\n')) {
          res += "  " + str;
          if (ss.rdbuf()->in_avail()) { res += "\n"; }
        }
      }
      if (idx == m_emitters.size() - 1) {
        res += "    \n";
      }
      else {
        res += "    ,\n";
      }
      idx++;
    }
  }
  res += "  ]\n";
  res += "}\n";
  return res;
}

hikari::assets::mitsuba::XMLScene::XMLScene()
{
}

auto hikari::assets::mitsuba::XMLSceneImporter::create(const std::string& filename) noexcept -> std::shared_ptr<XMLSceneImporter>
{
  return std::shared_ptr<XMLSceneImporter>(new XMLSceneImporter(filename));
}

hikari::assets::mitsuba::XMLSceneImporter::~XMLSceneImporter() noexcept
{
}

static auto loadXMLFloat(std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, const std::string& arr_name, void* p_val_elem) -> float {
  if (!p_val_elem) { return 0.0f; }
  auto val_elem = (tinyxml2::XMLElement*)p_val_elem;
  auto attr_val = val_elem->FindAttribute(arr_name.c_str());
  float res = 0.0f;
  if (attr_val) {
    scn::scan(context->normalizeString(attr_val->Value()), "{}", res);
  }
  return res;
}
static auto loadXMLArr(std::shared_ptr<hikari::assets::mitsuba::XMLContext>   context, const std::string& arr_name, void* p_val_elem, char splitter = ',') -> std::vector<float> {
  if (!p_val_elem) { return {}; }
  auto val_elem = (tinyxml2::XMLElement*)p_val_elem;
  auto attr_arr = val_elem->FindAttribute(arr_name.c_str());
  if (!attr_arr) { return {}; }
  auto attr_val = context->normalizeString(attr_arr->Value());
  auto res = std::vector<float>();
  auto tmp = 0.0f;
  std::stringstream ss;
  ss << attr_val;
  std::string sentence = "";
  while (std::getline(ss, sentence, splitter)) {
    (void)scn::scan(sentence, "{}", tmp);
    res.push_back(tmp);
  }
  return res;
}
static auto loadXMLVec3(std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, const std::string& arr_name, void* p_val_elem) -> glm::vec3{
  if (!p_val_elem) { return {}; }
  auto val_elem = (tinyxml2::XMLElement*)p_val_elem;
  auto arr      = loadXMLArr(context, arr_name,p_val_elem);
  if (!arr.empty()) {
    if (arr.size() == 1) { return glm::vec3(arr[0]); }
    if (arr.size() == 2) { return glm::vec3(arr[0], arr[1], 0.0f); }
    if (arr.size() >= 3) { return glm::vec3(arr[0], arr[1], arr[2]); }
    return {};
  }
  glm::vec3 res = {};
  auto attr_x   = val_elem->FindAttribute("x");
  auto attr_y   = val_elem->FindAttribute("y");
  auto attr_z   = val_elem->FindAttribute("z");
  if (attr_x) if (auto val = context->normalizeString(attr_x->Value()); !val.empty()) { if (scn::scan(val, "{}", res[0])) {}; }
  if (attr_y) if (auto val = context->normalizeString(attr_y->Value()); !val.empty()) { if (scn::scan(val, "{}", res[1])) {}; }
  if (attr_z) if (auto val = context->normalizeString(attr_z->Value()); !val.empty()) { if (scn::scan(val, "{}", res[2])) {}; }
  return res;
}
static auto loadXMLRgb (std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, const std::string& arr_name, void* p_val_elem) -> glm::vec3 {
  if (!p_val_elem) { return {}; }
  auto val_elem = (tinyxml2::XMLElement*)p_val_elem;
  auto arr = loadXMLArr(context, arr_name, p_val_elem);
  if (!arr.empty()) {
    if (arr.size() == 1) { return glm::vec3(arr[0]); }
    if (arr.size() == 2) { return glm::vec3(arr[0], arr[1], 0.0f); }
    if (arr.size() >= 3) { return glm::vec3(arr[0], arr[1], arr[2]); }
    return {};
  }
  glm::vec3 res = {};
  auto attr_r = val_elem->FindAttribute("r");
  auto attr_g = val_elem->FindAttribute("g");
  auto attr_b = val_elem->FindAttribute("b");
  if (auto val = context->normalizeString(attr_r->Value()); !val.empty()) { if (scn::scan(val, "{}", res[0])) {}; }
  if (auto val = context->normalizeString(attr_g->Value()); !val.empty()) { if (scn::scan(val, "{}", res[1])) {}; }
  if (auto val = context->normalizeString(attr_b->Value()); !val.empty()) { if (scn::scan(val, "{}", res[2])) {}; }
  return res;
}
static auto loadXMLMatrix(std::shared_ptr<hikari::assets::mitsuba::XMLContext>  context, const std::string& arr_name, void* p_val_elem) -> glm::mat4 {
  auto res = loadXMLArr(context, arr_name, p_val_elem, ' ');
  if (res.empty()) { return glm::mat4(1.0f); }
  if (res.size() == 9) {
    return glm::mat3(
      glm::vec3(res[0], res[3], res[6]),
      glm::vec3(res[1], res[4], res[7]),
      glm::vec3(res[2], res[5], res[8])
    );
  }
  if (res.size() == 16) {
    return glm::mat4(
      glm::vec4(res[0], res[4], res[8], res[12]),
      glm::vec4(res[1], res[5], res[9], res[13]),
      glm::vec4(res[2], res[6], res[10], res[14]),
      glm::vec4(res[3], res[7], res[11], res[15])
    );
  }return glm::mat4(1.0f);
}
static auto loadXMLRgbInline(std::shared_ptr<hikari::assets::mitsuba::XMLContext>  context, void* p_obj_elem) -> std::shared_ptr<hikari::assets::mitsuba::XMLSpectrum>
{
  auto obj_elem  = (tinyxml2::XMLElement*)p_obj_elem;
  if (!obj_elem) { return nullptr; }
  auto obj_value = obj_elem->Value();
  if (!obj_value) { return nullptr; }
  if ( obj_value != std::string_view("rgb")) { return nullptr; }
  auto attr_value = obj_elem->FindAttribute("value");
  if (!attr_value) { return nullptr; }
  auto spectrum   = hikari::assets::mitsuba::XMLSpectrum::create(context, "inline_rgb");
  spectrum->getProperties().setValue("value", loadXMLRgb(context, "value", obj_elem));
  return spectrum;
}
static auto loadXMLSpectrumInline(std::shared_ptr<hikari::assets::mitsuba::XMLContext>  context, void* p_obj_elem) -> std::shared_ptr<hikari::assets::mitsuba::XMLSpectrum>
{
  auto obj_elem = (tinyxml2::XMLElement*)p_obj_elem;
  if (!obj_elem) { return nullptr; }
  auto obj_value = obj_elem->Value();
  if (!obj_value) { return nullptr; }
  if (obj_value  != std::string_view("spectrum")) { return nullptr; }
  auto spectrum   = hikari::assets::mitsuba::XMLSpectrum::create(context,"inline_spectrum");
  auto attr_value = obj_elem->FindAttribute("value");
  if  (attr_value) {
    auto value = context->normalizeString(attr_value->Value());
    float tmp1; float tmp2;
    if (scn::scan(value, "{}:{}", tmp1, tmp2)) {
      spectrum->getProperties().setValue("value", std::string(value));
    }
    else {
      try {
        auto res = std::stof(value);
        spectrum->getProperties().setValue("value", res);
      }
      catch (std::exception&) {
        spectrum->getProperties().setValue("value", std::string(value));
      }
    }
  }
  auto attr_filename = obj_elem->FindAttribute("filename");
  if (attr_filename) {
    auto filename = context->normalizeString(attr_filename->Value());
    spectrum->getProperties().setValue("filename", filename);
  }
  return spectrum;
}
static auto loadXMLTransform(std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, void* p_obj_elem) -> std::shared_ptr<hikari::assets::mitsuba::XMLTransform>
{
  using namespace hikari::assets::mitsuba;
  auto obj_elem = (tinyxml2::XMLElement*)p_obj_elem;
  if (!obj_elem) { return nullptr; }
  auto obj_value = obj_elem->Value();
  if (!obj_value) { return nullptr; }
  if (obj_value != std::string_view("transform")) { return nullptr; }

  auto transform = XMLTransform::create(context);
  HK_MITSUBA_XML_FOR_EACH(prop_elem, obj_elem) {
    auto type_elem = prop_elem->Value();
    if (type_elem == std::string_view("translate")) {
      transform->addElement(XMLTransformElementTranslate::create(context,
        loadXMLVec3(context, "value", (void*)prop_elem))
      ); continue;
    }
    if (type_elem == std::string_view("rotate")) {
      transform->addElement(XMLTransformElementRotation::create(context,
        loadXMLVec3(context, "value", (void*)prop_elem),
        loadXMLFloat(context,"angle",(void*)prop_elem))
      ); continue;
    }
    if (type_elem == std::string_view("scale")) {
      transform->addElement(XMLTransformElementScale::create(context,
        loadXMLVec3(context, "value", (void*)prop_elem))
      ); continue;
    }
    if (type_elem == std::string_view("matrix")) {
      transform->addElement(XMLTransformElementMatrix::create(context,
        loadXMLMatrix(context, "value", (void*)prop_elem))
      ); continue;
    }
    if (type_elem == std::string_view("lookat")) {
      transform->addElement(XMLTransformElementLookAt::create(context,
        loadXMLVec3(context, "origin", (void*)prop_elem),
        loadXMLVec3(context, "target", (void*)prop_elem),
        loadXMLVec3(context, "up"    , (void*)prop_elem))
      ); continue;
    }

  }
  return transform;
}

auto hikari::assets::mitsuba::XMLSceneImporter::loadXMLObject(std::shared_ptr<hikari::assets::mitsuba::XMLContext> context, void* p_obj_elem) -> std::shared_ptr<XMLObject>
{
  auto obj_elem         = (tinyxml2::XMLElement*)p_obj_elem;
  if (!obj_elem ) { return nullptr; }
  auto obj_value        = obj_elem->Value();
  if (!obj_value) { return nullptr; }
  auto attr_plugin_type = obj_elem->FindAttribute("type");
  auto attr_id          = obj_elem->FindAttribute("id");
  if (!attr_plugin_type) { return nullptr; }
  auto plugin_type_name = context->normalizeString(attr_plugin_type->Value());
  auto id               = std::string("");
  if (attr_id) { id = context->normalizeString(attr_id->Value()); }
  auto object           = std::shared_ptr<XMLObject>();
  if (obj_value == std::string_view("bsdf"      )) { object = XMLBsdf      ::create(context, plugin_type_name, id); }
  if (obj_value == std::string_view("emitter"   )) { object = XMLEmitter   ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("film"      )) { object = XMLFilm      ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("integrator")) { object = XMLIntegrator::create(context, plugin_type_name); }
  if (obj_value == std::string_view("medium"    )) { object = XMLMedium    ::create(context, plugin_type_name, id); }
  if (obj_value == std::string_view("phase"     )) { object = XMLPhase     ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("rfilter"   )) { object = XMLRFilter   ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("sampler"   )) { object = XMLSampler   ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("sensor"    )) { object = XMLSensor    ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("shape"     )) { object = XMLShape     ::create(context, plugin_type_name, id); }
  if (obj_value == std::string_view("spectrum"  )) { object = XMLSpectrum  ::create(context, plugin_type_name); }
  if (obj_value == std::string_view("texture"   )) { object = XMLTexture   ::create(context, plugin_type_name, id); }
  if (obj_value == std::string_view("volume"    )) { object = XMLVolume    ::create(context, plugin_type_name); }
  {
    HK_MITSUBA_XML_FOR_EACH(prop_elem, obj_elem) {
      auto attr_name = prop_elem->FindAttribute("name");
      if (attr_name) {
        // named properties
        auto value = prop_elem->Value();
        auto name  = context->normalizeString(attr_name->Value());
        if (value  == std::string_view("integer")) {
          auto attrib_value = prop_elem->FindAttribute("value");
          if (!attrib_value) { throw std::runtime_error("Failed To Load Prop Int Val!"); }
          else {
            bool is_loaded = true;
            try {
              auto int_value = std::stoi(context->normalizeString(attrib_value->Value()));
              object->getProperties().setValue(name, int_value);
            }
            catch (std::exception&) { is_loaded = false; }
            if (!is_loaded) { throw std::runtime_error("Failed To Load Prop Int Val!"); }
          }
          continue;
        }
        if (value  == std::string_view("float")) {
          auto attrib_value = prop_elem->FindAttribute("value");
          if (!attrib_value) { throw std::runtime_error("Failed To Load Prop Flt Val!"); }
          else {
            bool is_loaded = true;
            try {
              auto flt_value = std::stof(context->normalizeString(attrib_value->Value()));
              object->getProperties().setValue(name, flt_value);
            }
            catch (std::exception&) { is_loaded = false; }
            if (!is_loaded) { throw std::runtime_error("Failed To Load Prop Flt Val!"); }
          }
          continue;
        }
        if (value  == std::string_view("string")) {
          auto attrib_value = prop_elem->FindAttribute("value");
          if (!attrib_value) { throw std::runtime_error("Failed To Load Prop Str Val!"); }
          else {
            bool is_loaded = true;
            try {
              auto str_value = context->normalizeString(attrib_value->Value());
              object->getProperties().setValue(name, str_value);
            }
            catch (std::exception&) { is_loaded = false; }
            if (!is_loaded) { throw std::runtime_error("Failed To Load Prop Str Val!"); }
          }
          continue;
        }
        if (value  == std::string_view("boolean")) {
          auto attrib_value = prop_elem->FindAttribute("value");
          if (!attrib_value) { throw std::runtime_error("Failed To Load Prop Bool Val!"); }
          else {
            bool is_loaded = true;
            try {
              bool bool_value = false;
              if (!scn::scan(context->normalizeString(attrib_value->Value()), "{}", bool_value)) {
                throw std::runtime_error("Failed To Load Prop Bool Val!");
              }
              else {
                object->getProperties().setValue(name, bool_value);
              }
            }
            catch (std::exception&) { is_loaded = false; }
            if (!is_loaded) { throw std::runtime_error("Failed To Load Prop Bool Val!"); }
          }
          continue;
        }
        if (value  == std::string_view("vector")) {
          object->getProperties().setValue(name, loadXMLVec3(context, "value", (void*)prop_elem));
          continue;
        }
        if (value  == std::string_view("point")) {
          object->getProperties().setValue(name, XMLPoint(loadXMLVec3(context, "value", (void*)prop_elem)));
          continue;
        }
        if (value  == std::string_view("ref")) {
          auto attrib_id = prop_elem->FindAttribute("id");
          if (!attrib_id) { throw std::runtime_error("Failed To Load Prop Ref!"); } else {
            object->getProperties().setValue("id", XMLRef{ context->normalizeString(attrib_id->Value()) });
          }
          continue;
        }
        if (value  == std::string_view("transform")) {
          auto transform = loadXMLTransform(context, (void*)prop_elem);
          if (!transform){ throw std::runtime_error("Failed To Load Prop Transform!"); }
          else {
            object->getProperties().setValue(name, transform);
          }
          continue;
        }
        if (value  == std::string_view("rgb")) {
          auto spectrum = loadXMLRgbInline(context, (void*)prop_elem);
          if (!spectrum) { throw std::runtime_error("Failed To Load Prop Rgb!"); }
          else {
            object->getProperties().setValue(name, spectrum);
          }
          continue;
        }
        if ((value == std::string_view("spectrum")) && !prop_elem->FindAttribute("type")) {
          auto spectrum = loadXMLSpectrumInline(context, (void*)prop_elem);
          if (!spectrum) { throw std::runtime_error("Failed To Load Prop Spectrum!"); }
          else {
            object->getProperties().setValue(name, spectrum);
          }
          continue;
        }
      }
      else {
        // nested properties
        auto value = prop_elem->Value();
        if (value == std::string_view("ref")) {
          auto attrib_id = prop_elem->FindAttribute("id");
          if (!attrib_id){ throw std::runtime_error("Failed To Load Nest Ref!"); } else {
            object->addNestRef(XMLRef{ context->normalizeString(attrib_id->Value()) });
          }
        }
        else {
          auto nest_obj = loadXMLObject(context, (void*)prop_elem);
          if (!nest_obj) { throw std::runtime_error("Failed To Load Nest Obj!"); }
          object->addNestObj(nest_obj);
        }
      }
    }
  }
  return object;
}

auto hikari::assets::mitsuba::XMLSceneImporter::loadChildScene(const XMLContextPtr& context, const std::string& filename) -> std::shared_ptr<XMLScene>
{
  if (!context) { return nullptr; }
  auto root_path = std::filesystem::path(context->getPath()).parent_path();
  for (auto& subpath : context->getSubPathes()) {
    auto path          = std::filesystem::canonical((root_path/ subpath) / filename);
    if (!std::filesystem::exists(path)) { continue; }
    auto scene_builder = XMLSceneImporter::create(path.string());
    auto scene         = scene_builder->loadScene();
    if (!scene) { continue; }
    else {
      auto relative_path = ((std::filesystem::path(subpath)) / filename).lexically_normal();
      scene->getContext()->setPath(relative_path.string());
      return scene;
    }
  }
  {
    auto path = std::filesystem::canonical(std::filesystem::path(root_path) / filename);
    if (!std::filesystem::exists(path)) {  return nullptr; }
    auto scene_builder = XMLSceneImporter::create(path.string());
    auto scene = scene_builder->loadScene();
    if (!scene) {  return nullptr;  }
    else {
      auto relative_path = ((std::filesystem::path(filename))).lexically_normal();
      scene->getContext()->setPath(relative_path.string());
      return scene;
    }
  }
  return nullptr;
}


auto hikari::assets::mitsuba::XMLSceneImporter::loadScene() -> std::shared_ptr<XMLScene>
{
  auto scene_xml_data = tinyxml2::XMLDocument();
  if (scene_xml_data.LoadFile(m_filename.c_str()) != tinyxml2::XML_SUCCESS) {
    return nullptr;
  }

  auto scene_xml_elem = scene_xml_data.FirstChildElement("scene");
  if (!scene_xml_elem) { return nullptr; }
  auto scene_xml_attr_version = scene_xml_elem->FindAttribute("version");
  if (!scene_xml_attr_version) { return nullptr; }
  auto scene_xml_attr_version_name = scene_xml_attr_version->Name();
  if (!scene_xml_attr_version_name) { return nullptr; }
  auto scene_xml_attr_version_value = scene_xml_attr_version->Value();
  if (!scene_xml_attr_version_value) { return nullptr; }

  int major, minor, patch;
  if (!scn::scan(std::string(scene_xml_attr_version_value), "{}.{}.{}", major, minor, patch)) {
    return nullptr;
  }

  auto context    = XMLContext::create(getFilename(), major, minor, patch);
  auto integrator = XMLIntegratorPtr();
  auto sensor     = XMLSensorPtr();
  auto shapes     = std::vector<XMLShapePtr>();
  auto bsdfs      = std::vector<XMLBsdfPtr>();
  auto emitters   = std::vector<XMLEmitterPtr>();

  HK_MITSUBA_XML_FOR_EACH(root_child, scene_xml_elem) {
    auto type_name = root_child->Value();
    if (type_name == std::string_view("default")) {
      auto attr_name = root_child->FindAttribute("name");
      auto attr_value = root_child->FindAttribute("value");
      if (attr_name && attr_value) {
        context->setDefValue(attr_name->Value(), attr_value->Value());
      }
      continue;
    }
    if (type_name == std::string_view("path")) {
      auto attr_value = root_child->FindAttribute("value");

      if (attr_value) { context->addSubPath(context->normalizeString(attr_value->Value())); }
      continue;
    }
    if (type_name == std::string_view("include")) {
      auto attr_filename = root_child->FindAttribute("filename");
      if (attr_filename) {
        auto filepath              = context->normalizeString(attr_filename->Value());
        auto child_scene           = loadChildScene(context, filepath);
        if (child_scene) {
          auto child_integrator    = child_scene->getIntegrator()  ;
          if (child_integrator) { integrator = child_integrator; }
          auto child_sensor        = child_scene->getSensor()      ;
          if (child_sensor)     { sensor = child_sensor; }
          auto child_emitters      = child_scene->getEmitters()    ;
          if (!child_emitters.empty()) {
            for (auto& emitter : child_emitters) {
              emitters.push_back(emitter);
             }
          }
          auto child_shapes        = child_scene->getShapes()      ;
          if (!child_shapes.empty()) {
            for (auto& shape   : child_shapes) {
              shapes.push_back(shape);
            }
          }
          auto child_context       = child_scene->getContext()     ;
          if (child_context) {  child_context->setParentContext(context); }
          auto child_search_paths  = child_context->getSubPathes();
          for (auto& child_search_path : child_search_paths) {
            auto child_path        = (std::filesystem::path(filepath).parent_path() / child_search_path).lexically_normal();
            context->addSubPath(child_path.string());
          }
          auto child_ref_aliases   = child_context->getAliasRefs() ;
          for (auto& [alias, id] : child_ref_aliases) {
            context->setAliasRef(alias, id);
          }
          auto child_ref_objects   = child_context->getRefObjects();
          for (auto& object : child_ref_objects) {
            context->setRefObject(object);
          }
        }
      }
      continue;
    }
    auto object    = loadXMLObject(context, (void*)root_child);
    if ( object) {
      if (object->getObjectType() == XMLObjectType::eIntegrator) {
        integrator = std::static_pointer_cast<XMLIntegrator>(object);
        continue;
      }
      if (object->getObjectType() == XMLObjectType::eSensor) {
        sensor     = std::static_pointer_cast<XMLSensor>(object);
        continue;
      }
      if (object->getObjectType() == XMLObjectType::eShape) {
        shapes.push_back(std::static_pointer_cast<XMLShape>(object));
        continue;
      }
      if (object->getObjectType() == XMLObjectType::eEmitter) {
        emitters.push_back(std::static_pointer_cast<XMLEmitter>(object));
        continue;
      }
    }
  }

  auto scene = XMLScene::create();
  scene->setContext(context);
  scene->setIntegrator(integrator);
  scene->setSensor(sensor);
  scene->setEmitters(emitters);
  scene->setShapes(shapes);
  return scene;
}

