#include <hikari/core/field.h>
#include <hikari/core/data_type.h>
#include <hikari/core/json.h>

auto hikari::core::Field::operator[](const Str& name) -> PropertyRef { auto object = getObject();  return PropertyRef(object, name); }

auto hikari::core::Field::operator[](const Str& name) const -> Property { return getValue(name); }

auto hikari::core::Field::operator[](size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); } return Field(object->getChild(idx)); }

auto hikari::core::Field::operator[](size_t idx) -> FieldRef { auto object = getObject(); return FieldRef(object, idx); }

void hikari::core::Field::setPropertyBlock(const PropertyBlock& pb)
{
  auto object = getObject();
  if (object) { object->setPropertyBlock(pb); }
}

void hikari::core::Field::getPropertyBlock(PropertyBlock& pb) const
{
  auto object = getObject();
  if (object) { object->getPropertyBlock(pb); }
}

auto hikari::core::Field::getSize() const -> size_t { return getChildCount(); }

void hikari::core::Field::setSize(size_t count) { setChildCount(count); }

auto hikari::core::Field::getName() const -> Str { auto object = getObject(); return object ? object->getName() : ""; }

void hikari::core::Field::setName(const Str& name) { auto object = getObject(); if (object) { object->setName(name); } }

auto hikari::core::Field::getObject() const -> std::shared_ptr<FieldObject> { return m_object; }

auto hikari::core::Field::getKeys() const -> std::vector<Str> { auto object = getObject(); if (object) { return object->getPropertyNames(); } else { return{}; } }

bool  hikari::core::Field::setValue(const Str& name, const Property& prop) { auto object = getObject(); if (!object) { return false; } return object->setProperty(name, prop); }

bool  hikari::core::Field::getValue(const Str& name, Property& prop) const { auto object = getObject(); if (!object) { return false; } return object->getProperty(name, prop); }

auto hikari::core::Field::getValue(const Str& name) const -> Property { auto object = getObject(); if (!object) { return Property(); } Property res; object->getProperty(name, res); return res; }

bool  hikari::core::Field::hasValue(const Str& name) const { auto object = getObject();  if (!object) { return false; } return object->hasProperty(name); }

auto hikari::core::Field::getChildCount() const -> size_t { auto object = getObject(); return object->getChildCount(); }

void hikari::core::Field::setChildCount(size_t count) { auto object = getObject(); object->setChildCount(count); }

auto hikari::core::Field::getChildren() const -> std::vector<Field> {
  auto res = std::vector<Field>();
  auto object = getObject();
  if (object) {
    for (auto i = 0; i < object->getChildCount(); ++i) {
      res.push_back(Field(object->getChild(i)));
    }
  }
  return res;
}

void hikari::core::Field::setChildren(const std::vector<Field>& children) {
  auto object = getObject();
  if (!object) { return; }
  std::vector<std::shared_ptr<FieldObject>> field_objects = {};
  for (auto& child : children) {
    field_objects.push_back(child.getObject());
  }
  object->setChildren(field_objects);
}

void hikari::core::Field::popChildren() {
  auto object = getObject();
  if (!object) { return; }
  object->popChildren();
}

auto hikari::core::Field::getChild(size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); }return Field(object->getChild(idx)); }

void hikari::core::Field::setChild(size_t idx, const Field& child) { auto object = getObject(); if (!object) { return; } object->setChild(idx, child.getObject()); }

void hikari::core::Field::addChild(const Field& field) { auto object = getObject(); if (!object) { return; } object->addChild(field.getObject()); }

void hikari::core::Field::popChild(size_t idx) { auto object = getObject(); if (!object) { return; } object->popChild(idx); }

void hikari::core::FieldRef::operator=(const Field& field) noexcept {
  auto obj = m_object.lock();
  if (obj) {
    obj->setChild(m_idx, field.getObject());
  }
}

auto hikari::core::FieldRef::operator[](const Str& name) -> PropertyRef { auto object = getObject();   return PropertyRef(object, name); }

auto hikari::core::FieldRef::operator[](const Str& name) const -> Property { return getValue(name); }

auto hikari::core::FieldRef::operator[](size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); } return Field(object->getChild(idx)); }

auto hikari::core::FieldRef::operator[](size_t idx) -> Ref { auto object = getObject(); return Ref(object, idx); }

void hikari::core::FieldRef::setPropertyBlock(const PropertyBlock& pb)
{
  auto object = getObject();
  if (object) { object->setPropertyBlock(pb); }
}

void hikari::core::FieldRef::getPropertyBlock(PropertyBlock& pb) const
{
  auto object = getObject();
  if (object) { object->getPropertyBlock(pb); }
}

auto hikari::core::FieldRef::getSize() const -> size_t { return getChildCount(); }

void hikari::core::FieldRef::setSize(size_t count) { setChildCount(count); }

auto hikari::core::FieldRef::getName() const -> Str { auto object = getObject(); return object ? object->getName() : ""; }

void hikari::core::FieldRef::setName(const Str& name) { auto object = getObject(); if (object) { object->setName(name); } }

auto hikari::core::FieldRef::getObject() const -> std::shared_ptr<FieldObject> {
  auto object = m_object.lock();
  if (!object) { return nullptr; }
  return object->getChild(m_idx);
}

auto hikari::core::FieldRef::getKeys() const -> std::vector<Str> { auto object = getObject(); if (object) { return object->getPropertyNames(); } else { return{}; } }

bool   hikari::core::FieldRef::setValue(const Str& name, const Property& prop) { auto object = getObject(); if (!object) { return false; } return object->setProperty(name, prop); }

bool   hikari::core::FieldRef::getValue(const Str& name, Property& prop) const { auto object = getObject(); if (!object) { return false; } return object->getProperty(name, prop); }

auto hikari::core::FieldRef::getValue(const Str& name) const -> Property { auto object = getObject(); if (!object) { return Property(); } Property res; object->getProperty(name, res); return res; }

bool   hikari::core::FieldRef::hasValue(const Str& name) const { auto object = getObject();  if (!object) { return false; } return object->hasProperty(name); }

auto hikari::core::FieldRef::getChildCount() const -> size_t { auto object = getObject(); return object->getChildCount(); }

void hikari::core::FieldRef::setChildCount(size_t count) { auto object = getObject(); object->setChildCount(count); }

auto hikari::core::FieldRef::getChildren() const -> std::vector<Field> {
  auto res = std::vector<Field>();
  auto object = getObject();
  if (object) {
    for (auto i = 0; i < object->getChildCount(); ++i) {
      res.push_back(Field(object->getChild(i)));
    }
  }
  return res;
}

void hikari::core::FieldRef::setChildren(const std::vector<Field>& children) {
  auto object = getObject();
  if (!object) { return; }
  std::vector<std::shared_ptr<FieldObject>> field_objects = {};
  for (auto& child : children) {
    field_objects.push_back(child.getObject());
  }
  object->setChildren(field_objects);
}

void hikari::core::FieldRef::popChildren() {
  auto object = getObject();
  if (!object) { return; }
  object->popChildren();
}

auto hikari::core::FieldRef::getChild(size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); }return Field(object->getChild(idx)); }

void hikari::core::FieldRef::setChild(size_t idx, const Field& child) { auto object = getObject(); if (!object) { return; } object->setChild(idx, child.getObject()); }

void hikari::core::FieldRef::addChild(const Field& field) { auto object = getObject(); if (!object) { return; } object->addChild(field.getObject()); }

void hikari::core::FieldRef::popChild(size_t idx) { auto object = getObject(); if (!object) { return; } object->popChild(idx); }

auto hikari::core::FieldObject::create(Str name) -> std::shared_ptr<FieldObject> {
  return std::shared_ptr<FieldObject>(new FieldObject(name));
}

hikari::core::FieldObject::~FieldObject() noexcept {}

auto hikari::core::FieldObject::getJSONString() const -> Str {
  Str res = "";
  res += "\{";
  res += " \"name\" : \"" + m_name + "\" ,";
  res += " \"type\" : \"Field\" ,";
  res += " \"properties\" : {";
  res += " \"children\": [";
  {
    auto children = getChildren();
    for (auto i = 0; i < children.size(); ++i) {
      res += " " + children[i]->getJSONString();
      if (i != children.size() - 1) { res += " ,"; }
    }
  }
  res += "]";
  {
    auto keys = getPropertyNames();
    if (keys.size() > 1) {
      res += " ,";
      size_t i = 0;
      for (auto& key : keys) {
        if (key != "children") {
          auto prop = Object::getProperty(key);
          auto str = prop.getJSONString();
          res += "\"" + key + "\" : " + str;
          if (i != keys.size() - 2) { res += " ,"; }
          ++i;
        }
      }
    }
  }
  res += " }";
  res += "\}";
  return res;
}

auto hikari::core::FieldObject::getName() const -> Str { return m_name; }

void hikari::core::FieldObject::setName(const Str& name) { m_name = name; }

auto hikari::core::FieldObject::getPropertyNames() const -> std::vector<Str> {
  auto strs = m_property_block.getKeys();
  if (std::find(strs.begin(), strs.end(), "children") == strs.end()) {
    strs.push_back("children");
  }
  return strs;
}

void hikari::core::FieldObject::setPropertyBlock(const PropertyBlock& pb) {
  m_name = "";
  m_children = {};
  m_property_block = {};
  auto keys = pb.getKeys();
  for (auto& key : keys) {
    auto value = pb.getValue(key);
    if (key == "children") {
      if (value.getTypeIndex() == PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value) {
        auto children = value.getValue<std::vector<std::shared_ptr<Object>>>();
        bool field_child = true;
        for (auto& child : children) {
          if (child->getTypeString() != "Field") { field_child = false; }
        }
        if (field_child) {
          for (auto& child : children) {
            m_children.push_back(std::static_pointer_cast<FieldObject>(child));
          }
        }
      }
      else {
        m_property_block.setValue(key, value);
      }
    }
    else {
      m_property_block.setValue(key, value);
    }
  }
}

void hikari::core::FieldObject::getPropertyBlock(PropertyBlock& pb) const {
  pb = m_property_block;
  auto children = std::vector<std::shared_ptr<Object>>();
  for (auto& child : m_children) {
    children.push_back(child);
  }
  pb.setValue("children", children);
}

bool hikari::core::FieldObject::hasProperty(const Str& name) const {
  if (name == "children") { return true; }
  return m_property_block.hasValue(name);
}

bool hikari::core::FieldObject::setProperty(const Str& name, const Property& value) {
  if (name == "children") {
    if (value.getTypeIndex() == PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value)
    {
      auto children = value.getValue<std::vector<std::shared_ptr<Object>>>();
      bool field_child = true;
      for (auto& child : children) {
        if (child->getTypeString() != "Field") { field_child = false; }
      }
      m_children = {};
      if (field_child) {
        for (auto& child : children) {
          m_children.push_back(std::static_pointer_cast<FieldObject>(child));
        }
      }
      return true;
    }
    else {
      return false;
    }
  }
  else {
    m_property_block.setValue(name, value);
    return true;
  }
}

bool hikari::core::FieldObject::getProperty(const Str& name, Property& value) const {
  value = m_property_block.getValue(name);
  if (name == "children") {
    std::vector<std::shared_ptr<Object>> objects = {};
    for (auto& child : m_children) { objects.push_back(child); }
    value = objects;
    return true;
  }
  else {
    return m_property_block.getValue(name, value);
  }
}

bool hikari::core::FieldObject::getPropertyTypeIndex(const Str& name, size_t& type_index) const {
  if (name == "children") { type_index = PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value; return true; }
  return m_property_block.getTypeIndex(name, type_index);
}

auto hikari::core::FieldObject::getChildCount() const -> size_t { return m_children.size(); }

void hikari::core::FieldObject::setChildCount(size_t count) {
  size_t old_size = m_children.size();
  m_children.resize(count, nullptr);
  if (old_size < count) {
    for (size_t i = old_size; i < count; ++i) {
      m_children[i] = FieldObject::create("");
    }
  }
}

auto hikari::core::FieldObject::getChildren() const -> std::vector<std::shared_ptr<FieldObject>> { return m_children; }

void hikari::core::FieldObject::setChildren(const std::vector<std::shared_ptr<FieldObject>>& children) { m_children = children; }

void hikari::core::FieldObject::popChildren() { m_children = {}; }

auto hikari::core::FieldObject::getChild(size_t idx) const -> std::shared_ptr<FieldObject> { return m_children[idx]; }

void hikari::core::FieldObject::setChild(size_t idx, std::shared_ptr<FieldObject> child) { if (m_children.size() > idx) { m_children[idx] = child; } }

void hikari::core::FieldObject::addChild(std::shared_ptr<FieldObject> child) { m_children.push_back(child); }

void hikari::core::FieldObject::popChild(size_t idx) { if (m_children.size() > idx) { m_children.erase(m_children.begin() + idx); } }

auto hikari::core::convertToJSONString(const core::Field& v) -> Str {
  auto json = convertFieldToJSON(v);
  return json.dump();
}

auto hikari::core::convertJSONStringToField(const Str& str) -> Field
{
  return convertJSONToField(nlohmann::json::parse(str));
}

auto hikari::core::convertJSONToField(const Json& json) -> Field
{
  if (json.is_object()) {
    auto iter_type = json.find("type");
    auto iter_name = json.find("name");
    auto iter_properties = json.find("properties");
    if (iter_type == json.end()) { return Field(nullptr); }
    if (iter_name == json.end()) { return Field(nullptr); }
    if (iter_properties == json.end()) { return Field(nullptr); }
    if (!iter_type.value().is_string()) { return Field(nullptr); }
    if (!iter_name.value().is_string()) { return Field(nullptr); }
    if (!iter_properties.value().is_object()) { return Field(nullptr); }
    if (iter_type.value().get<std::string>() != "Field") { return Field(nullptr); }
    auto& properties = iter_properties.value();
    auto iter_children = properties.find("children");
    if (iter_children == properties.end()) { return Field(nullptr); }
    if (!iter_children.value().is_array()) { return Field(nullptr); }
    auto field = Field(iter_name.value().get<std::string>());
    for (auto& child : iter_children.value()) { field.addChild(convertJSONToField(child)); }
    for (auto& elem : properties.items()) {
      if (elem.key() == "children") { continue; }
      auto prop = convertJSONStringToProperty(elem.value().dump());
      if (prop) {
        field.setValue(elem.key(), prop);
      }
    }
    return field;
  }
  else {
    return Field(nullptr);
  }
}

auto hikari::core::convertFieldToJSON(const Field& field)-> Json
{
  return Json::parse(field.getJSONString());
}
