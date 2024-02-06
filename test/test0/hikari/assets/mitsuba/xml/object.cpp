#include "object.h"
#include <sstream>
#include <fmt/format.h>

#define HK_OBJECT_IMPL_CONCAT(IN1, IN2)               HK_OBJECT_IMPL_CONCAT_IMPL(IN1, IN2)
#define HK_OBJECT_IMPL_CONCAT_IMPL(IN1, IN2)          IN1##IN2
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Boolean   boolean
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Integer   integer
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Float     float
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_String    string
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Vector    vector
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Point     point
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Ref       ref
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_Object    object
#define HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(UPPERCASE) HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE_,UPPERCASE)
#define HK_OBJECT_IMPL_FOR_EACH_TYPES(MACRO)            HK_OBJECT_IMPL_FOR_EACH_TYPES_IMPL(MACRO) 
#define HK_OBJECT_IMPL_FOR_EACH_TYPES_IMPL(MACRO)       \
MACRO(Boolean  );\
MACRO(Integer  );\
MACRO(Float    );\
MACRO(String   );\
MACRO(Vector   );\
MACRO(Point    );\
MACRO(Ref      );\
MACRO(Object   )
#define HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(MACRO)            HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES_IMPL(MACRO) 
#define HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES_IMPL(MACRO)       \
MACRO(Boolean  );\
MACRO(Integer  );\
MACRO(Float    );\
MACRO(String   );\
MACRO(Vector   );\
MACRO(Point    );\
MACRO(Ref      )


hikari::assets::mitsuba::XMLProperties::XMLProperties() noexcept
{
}

hikari::assets::mitsuba::XMLProperties::~XMLProperties() noexcept
{
}

bool hikari::assets::mitsuba::XMLProperties::getValueType(const std::string& name, Type& type) const noexcept {
  auto iter = m_types.find(name);
  if (iter != std::end(m_types)) { type = iter->second; return true; }
  else { return false; }
}

auto hikari::assets::mitsuba::XMLProperties::getValueType(const std::string& name) const noexcept -> std::optional<Type>
{
  auto iter = m_types.find(name);
  if (iter != std::end(m_types)) { return iter->second; }
  else { return std::nullopt; }
}

bool hikari::assets::mitsuba::XMLProperties::hasValue(const std::string& name) const {
  return m_types.count(name) > 0;
}

bool hikari::assets::mitsuba::XMLProperties::hasValue(const std::string& name, Type type) const {
  auto iter = m_types.find(name);
  if (iter != std::end(m_types)) { return type == iter->second; }
  else { return false; }
}

bool hikari::assets::mitsuba::XMLProperties::getValue(const std::string& name, XMLProperty& prop) const {
  Type type = {};
  if (!getValueType(name, type)) { return false; }
  switch (type) {
#define HK_OBJECT_IMPL_XML_PROPERTIES_GET_VALUE_CASE(TYPE) \
  case HK_OBJECT_IMPL_CONCAT(Type::e,TYPE): { prop.setValue(HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.at(name)));} break
    
  HK_OBJECT_IMPL_FOR_EACH_TYPES(HK_OBJECT_IMPL_XML_PROPERTIES_GET_VALUE_CASE);
  }
  return true;
}

auto hikari::assets::mitsuba::XMLProperties::getValue(const std::string& name) const -> std::optional<XMLProperty>
{
  XMLProperty prop;
  if (getValue(name, prop)) { return prop; }
  else { return std::nullopt; }
}

#define HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL(TYPE)                  \
void hikari::assets::mitsuba::XMLProperties::setValue(const std::string& name, HK_OBJECT_IMPL_CONCAT(XML,TYPE) value) noexcept { \
  eraseValue(name); m_types[name] = HK_OBJECT_IMPL_CONCAT(Type::e,TYPE); HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.insert({ name,value })); \
}

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL);

void hikari::assets::mitsuba::XMLProperties::setValue(const std::string& name, std::shared_ptr<XMLObject> value) noexcept
{
  if (!value) { return; }
  eraseValue(name);
  m_types[name] = Type::eObject;
  m_value_objects[name] = value;
}

void hikari::assets::mitsuba::XMLProperties::setValue(const std::string& name, const XMLProperty& value) noexcept {
  eraseValue(name);
  auto valueType = value.getType();
  m_types[name] = valueType;
  switch (valueType)
  {
#define HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL_VALUE_TYPE_CASE(TYPE)              \
case HK_OBJECT_IMPL_CONCAT(XMLProperty::Type::e,TYPE): { auto tmp_value = HK_OBJECT_IMPL_CONCAT(value.get,TYPE)();HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.insert({ name,*tmp_value })); } break
#define HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL_OBJECT_TYPE_CASE(TYPE)              \
case HK_OBJECT_IMPL_CONCAT(XMLProperty::Type::e,TYPE): { auto tmp_value = HK_OBJECT_IMPL_CONCAT(value.get,TYPE)();HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.insert({ name, tmp_value })); } break

  HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES( HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL_VALUE_TYPE_CASE);
  HK_OBJECT_IMPL_XML_PROPERTIES_SET_VALUE_IMPL_OBJECT_TYPE_CASE(Object);
  default:
    break;
  }
}

#define HK_ASSETS_MITSUBA_XML_OBJECT_PROPERTIES_GET_VALUE_IMPL(TYPE)                                                                                           \
bool hikari::assets::mitsuba::XMLProperties::getValue(const std::string& name, HK_OBJECT_IMPL_CONCAT(XML,TYPE)& value) const noexcept {                        \
  auto iter = HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.find(name));                                   \
  if (iter != HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.end())) { value = iter->second; return true; } \
  return false;                                                                                                                                                \
}

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_ASSETS_MITSUBA_XML_OBJECT_PROPERTIES_GET_VALUE_IMPL);

bool hikari::assets::mitsuba::XMLProperties::getValue(const std::string& name, std::shared_ptr<XMLObject>& value) const noexcept
{
  Type valueType = {};
  if (!getValueType(name, valueType)) { return false; }
#define HK_OBJECT_IMPL_XML_PROPERTIES_GET_VALUE_IMPL_CASE(TYPE)              \
  if (valueType == HK_OBJECT_IMPL_CONCAT(Type::e,TYPE)) { value =HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.at(name)); return true; }

  HK_OBJECT_IMPL_XML_PROPERTIES_GET_VALUE_IMPL_CASE(Object);
  return false;
}


#define HK_OBJECT_IMPL_XML_PROPERTIES_GET_TYPE_IMPL_VALUE_TYPE(TYPE)        \
  auto hikari::assets::mitsuba::XMLProperties:: HK_OBJECT_IMPL_CONCAT(get,TYPE) (const std::string& name) const noexcept -> std::optional<  HK_OBJECT_IMPL_CONCAT(XML,TYPE) > { \
  HK_OBJECT_IMPL_CONCAT(XML,TYPE) value; if (getValue(name, value)) { return value; } \
  else { return std::nullopt; } \
}                                                                     \

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_OBJECT_IMPL_XML_PROPERTIES_GET_TYPE_IMPL_VALUE_TYPE);

auto hikari::assets::mitsuba::XMLProperties::getObject(const std::string& name) const noexcept -> std::shared_ptr<XMLObject> {
  Type valueType;
  if (!getValueType(name, valueType)) { return nullptr; }
  if (valueType != Type::eObject) { return nullptr; }
  return m_value_objects.at(name);
}


hikari::assets::mitsuba::XMLObject::XMLObject(Type object_type, const std::string& plugin_type) :
  m_object_type{ object_type }, m_plugin_type{ plugin_type }
{}
hikari::assets::mitsuba::XMLObject::~XMLObject() {}

auto hikari::assets::mitsuba::XMLObject::toStrings() const -> std::vector<std::string>
{
  using namespace std::string_literals;
  auto res = std::vector<std::string>();
  res.push_back(fmt::format("\"object_type\":\"{}\"", getObjectTypeString()));
  res.push_back(fmt::format("\"plugin_type\":\"{}\"", m_plugin_type));
  {
    auto prop_res = std::string("");
    prop_res += "\"properties\": {\n";
    auto strs = m_properties.toStrings();
    auto idx = 0;
    for (auto& str : strs) {
      std::stringstream ss;
      ss << str;
      std::string sentence;
      while (std::getline(ss, sentence, '\n')) {
        prop_res += "  " + sentence;
        if (ss.rdbuf()->in_avail()) { prop_res += "\n"; }
      }
      if (idx == strs.size() - 1) { prop_res += "\n"; }
      else { prop_res += ",\n"; }
      ++idx;
    }
    prop_res += "}\n";
    res.push_back(prop_res);
  }
  {
    auto objs = getNestObjects();
    if (!objs.empty()) {
      auto nest_obj = std::string("");
      nest_obj += "\"nest_ref_objects\": [\n";
      auto idx = 0;
      for (auto& obj : objs) {
        auto str = obj->toString();
        std::stringstream ss; ss << str;
        std::string sentence;
        while (std::getline(ss, sentence, '\n')) {
          nest_obj += "  " + sentence;
          if (ss.rdbuf()->in_avail()) { nest_obj += "\n"; }
        }
        if (idx != objs.size() - 1) { nest_obj += "  ,\n"; }
        else { nest_obj += "  \n"; }
        ++idx;
      }
      nest_obj += "]\n";
      res.push_back(nest_obj);
    }
  }
  if (!m_nest_refs.empty()){
    auto nest_res = std::string("");
    nest_res += "\"nest_ref_indices\": [\n";
    auto idx = 0;
    for (auto& ref : m_nest_refs) {
      nest_res += "  \"" + ref.id + "\"";
      if (idx == m_nest_refs.size() - 1) { nest_res += "\n"; }
      else { nest_res += ",\n"; }
      ++idx;
    }
    nest_res += "]\n";
    res.push_back(nest_res);
  }
  return res;
}

auto hikari::assets::mitsuba::XMLObject::toString() const -> std::string
{
  using namespace std::string_literals;
  auto res = std::string();
  res += "{\n";
  auto tmps = toStrings();
  auto idx = 0;
  for (auto& tmp : tmps) {
    std::stringstream ss;
    ss << tmp;
    std::string sentence;
    while (std::getline(ss, sentence, '\n')) {
      res += "  "s + sentence;
      if (ss.rdbuf()->in_avail()) {
        res += +"\n";
      }
    }
  if (idx == tmps.size() - 1) {
  res += "\n";
  }
  else {
  res += ",\n";
  }
   ++idx;
  }
  res += "}\n";
  return res;
}

void hikari::assets::mitsuba::XMLProperties::eraseValue(const std::string& name)
{
  Type valueType = {};
  if (!getValueType(name, valueType)) { return; }

#define HK_OBJECT_IMPL_XML_OBJECT_ERASE_IMPL_CASE(TYPE) case Type:: HK_OBJECT_IMPL_CONCAT(e,TYPE): { HK_OBJECT_IMPL_CONCAT(HK_OBJECT_IMPL_CONCAT(m_value_,HK_OBJECT_IMPL_CONVERT_TO_LOWERCASE(TYPE)),s.erase(name)); } break
  switch (valueType)
  {
    HK_OBJECT_IMPL_FOR_EACH_TYPES(HK_OBJECT_IMPL_XML_OBJECT_ERASE_IMPL_CASE);
  default:
    break;
  }
}

auto hikari::assets::mitsuba::XMLProperties::toString() const -> std::string
{
  using namespace std::string_literals;
  std::string res = "";
  res +="{\n";
  auto idx = 0;
  auto tmps = toStrings();
  for (auto& tmp : tmps) {
    auto sentence = std::string("");
    std::stringstream ss;
    ss << tmp;
    while (std::getline(ss, sentence, '\n')) {
      res += "  " + sentence;
      if (ss.rdbuf()->in_avail()) { res += "\n"; }
    }
    if (idx == tmps.size() - 1) {
      res += "\n";
    }
    else {
      res += ",\n";
    }
    idx++;
  }
  res +="}\n";
  return res;
}

auto hikari::assets::mitsuba::XMLProperties::toStrings() const -> std::vector<std::string>
{
  using namespace std::string_literals;
  std::vector<std::string> res = {};
  for (auto& [name, value] : m_value_booleans) {
    res.push_back("\"" + name + "\" : "s + (value ? "true"s : "false"s));
  }
  for (auto& [name, value] : m_value_integers) {
    res.push_back("\"" + name + "\" : "s + std::to_string(value));
  }
  for (auto& [name, value] : m_value_floats) {
    res.push_back("\"" + name + "\" : "s + std::to_string(value));
  }
  for (auto& [name, value] : m_value_strings) {
    res.push_back("\"" + name + "\" : \""s + value + "\"");
  }
  for (auto& [name, value] : m_value_vectors) {
    res.push_back("\"" + name + "\" : ["s + fmt::format("{},{},{}", value.x, value.y, value.z) + "]");
  }
  for (auto& [name, value] : m_value_points) {
    res.push_back("\"" + name + "\" : ["s + fmt::format("{},{},{}", value.value.x, value.value.y, value.value.z) + "]");
  }
  for (auto& [name, value] : m_value_objects) {
    auto strs = value->toStrings();
    std::string tmp_res = "\"" + name + "\" :{\n";
    auto idx = 0;
    for(auto& str:strs) {
      std::stringstream ss;
      ss << str;
      std::string sentence;
      while (std::getline(ss, sentence, '\n')) {
        tmp_res += "  " + sentence ;
        if (ss.rdbuf()->in_avail()) {
          tmp_res += "\n";
        }
      }
      if (idx == strs.size() - 1) { tmp_res += "\n"; }
      else { tmp_res += ",\n"; }
    ++idx;
    }
    tmp_res += "}";
    res.push_back(tmp_res);
  }
  return res;
}

auto hikari::assets::mitsuba::XMLObject::getObjectType() const -> Type { return m_object_type; }

auto hikari::assets::mitsuba::XMLObject::getObjectTypeString() const -> std::string
{
  switch (m_object_type)
  {
  case Type::eBsdf: { return "bsdf"; }
  case Type::eEmitter: { return "emitter"; }
  case Type::eFilm: { return "film"; }
  case Type::eIntegrator: { return "integrator"; }
  case Type::eMedium: { return "medium"; }
  case Type::ePhase: { return "phase"; }
  case Type::eRFilter: { return "rfilter"; }
  case Type::eSampler: { return "sampler"; }
  case Type::eSensor: { return "sensor"; }
  case Type::eShape: { return "shape"; }
  case Type::eSpectrum: { return "spectrum"; }
  case Type::eTexture: { return "texture"; }
  case Type::eTransform: { return "transform"; }
  case Type::eVolume: { return "volume"; }
  default:
    return "Unknown";
    break;
  }
}

auto hikari::assets::mitsuba::XMLObject::getPluginType() const -> std::string { return m_plugin_type; }

auto hikari::assets::mitsuba::XMLObject::getProperties() const -> const XMLProperties& { return m_properties; }

auto hikari::assets::mitsuba::XMLObject::getProperties() -> XMLProperties& { return m_properties; }

auto hikari::assets::mitsuba::XMLObject::getNestRefCount() const noexcept -> size_t
{
  return m_nest_refs.size();
}

void hikari::assets::mitsuba::XMLObject::setNestRefCount(size_t count) noexcept
{
  m_nest_refs.resize(count, XMLRef(""));
  m_nest_refs.shrink_to_fit();
}

auto hikari::assets::mitsuba::XMLObject::getNestRef(size_t idx) const noexcept -> XMLRef
{
  if (idx < m_nest_refs.size()) { return m_nest_refs.at(idx); }
  else { return XMLRef(""); }
}

void hikari::assets::mitsuba::XMLObject::setNestRef(size_t idx, const XMLRef& ref) noexcept
{
  auto count = getNestRefCount();
  if (idx >= count) {
    setNestRefCount(count);
  }
  m_nest_refs[idx] = ref;
}

void hikari::assets::mitsuba::XMLObject::addNestRef(const XMLRef& ref) noexcept
{
  m_nest_refs.push_back(ref);
}

auto hikari::assets::mitsuba::XMLObject::getNestObjects() const noexcept -> std::vector<std::shared_ptr<XMLObject>>
{
  auto res= std::vector<std::shared_ptr<XMLObject>>();
  for (auto& objs : m_nest_objs) {
    for (auto& obj : objs) {
      if (obj) res.push_back(obj);
    }
  }
  return res;
}

auto hikari::assets::mitsuba::XMLObject::getNestObjCount(Type objectType) const noexcept -> size_t
{
  size_t idx = (size_t)objectType;
  if (idx < (size_t)Type::eCount) { return m_nest_objs[idx].size(); }
  return 0u;
}

void hikari::assets::mitsuba::XMLObject::setNestObjCount(Type objectType, size_t count) noexcept
{
  m_nest_objs[(int)objectType].resize(count, nullptr);
  m_nest_objs[(int)objectType].shrink_to_fit();
}

auto hikari::assets::mitsuba::XMLObject::getNestObj(Type objectType, size_t idx) const noexcept -> std::shared_ptr<XMLObject>
{
  auto count = getNestObjCount(objectType);
  if (idx < count) { return m_nest_objs[(int)objectType][idx]; }
  return nullptr;
}

void hikari::assets::mitsuba::XMLObject::setNestObj(size_t idx, const std::shared_ptr<XMLObject>& object) noexcept
{
  if (!object) { return; }
  auto objectType = object->getObjectType();
  auto count = getNestObjCount(objectType);
  if (idx >= count){
    setNestObjCount(objectType, idx + 1);
  }
  m_nest_objs[(int)objectType][idx] = object;
}

void hikari::assets::mitsuba::XMLObject::addNestObj(const std::shared_ptr<XMLObject>& object) noexcept
{
  if (!object) { return; }
  auto objectType = object->getObjectType();
  m_nest_objs[(int)objectType].push_back(object);
}

hikari::assets::mitsuba::XMLProperty::XMLProperty()  noexcept {}

hikari::assets::mitsuba::XMLProperty::~XMLProperty() noexcept {}

auto hikari::assets::mitsuba::XMLProperty::getType() const noexcept -> Type { return m_type; }

#define HK_OBJECT_IMPL_XML_PROPERTY_SET_VALUE_IMPL_VALUE_TYPE(TYPE) \
  void hikari::assets::mitsuba::XMLProperty::setValue( HK_OBJECT_IMPL_CONCAT(XML,TYPE) value) noexcept { m_type = Type::  HK_OBJECT_IMPL_CONCAT(e,TYPE); m_value = value; }

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_OBJECT_IMPL_XML_PROPERTY_SET_VALUE_IMPL_VALUE_TYPE);

void hikari::assets::mitsuba::XMLProperty::setValue(std::shared_ptr<XMLObject> value) noexcept {
  if (!value) { return; }
  m_value = value;
}

#define HK_OBJECT_IMPL_XML_PROPERTY_GET_VALUE_IMPL_VALUE_TYPE(TYPE) \
  bool hikari::assets::mitsuba::XMLProperty::getValue( HK_OBJECT_IMPL_CONCAT(XML,TYPE)& value) const noexcept { if (m_type == Type:: HK_OBJECT_IMPL_CONCAT(e,TYPE)) { value = std::get< HK_OBJECT_IMPL_CONCAT(XML,TYPE) >(m_value); return true; } return false; }

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_OBJECT_IMPL_XML_PROPERTY_GET_VALUE_IMPL_VALUE_TYPE);

bool hikari::assets::mitsuba::XMLProperty::getValue(std::shared_ptr<XMLObject>& value) const noexcept { auto ptr = std::get_if<std::shared_ptr<XMLObject>>(&m_value); if (ptr) { value = *ptr; return true; } return false; }

#define HK_OBJECT_IMPL_XML_PROPERTY_GET_TYPE_IMPL_VALUE_TYPE(TYPE) \
auto hikari::assets::mitsuba::XMLProperty:: HK_OBJECT_IMPL_CONCAT(get,TYPE)() const noexcept -> std::optional< HK_OBJECT_IMPL_CONCAT(XML,TYPE) > { \
  if (m_type == Type:: HK_OBJECT_IMPL_CONCAT(e,TYPE) ) { return std::get< HK_OBJECT_IMPL_CONCAT(XML,TYPE) >(m_value); } \
  return std::nullopt; \
}

HK_OBJECT_IMPL_FOR_EACH_VALUE_TYPES(HK_OBJECT_IMPL_XML_PROPERTY_GET_TYPE_IMPL_VALUE_TYPE);

auto hikari::assets::mitsuba::XMLProperty::getObject() const noexcept -> std::shared_ptr<XMLObject> {
  auto ptr = std::get_if<std::shared_ptr<XMLObject>>(&m_value); if (ptr) { return *ptr; } return nullptr;
}

hikari::assets::mitsuba::XMLReferableObject::~XMLReferableObject() noexcept {}

auto hikari::assets::mitsuba::XMLReferableObject::getID() const -> std::string
{
  return m_id;
}

hikari::assets::mitsuba::XMLReferableObject::XMLReferableObject(Type object_type, const std::string& plugin_type, const std::string& id)
  :XMLObject(object_type,plugin_type),m_id{id}
{
}

