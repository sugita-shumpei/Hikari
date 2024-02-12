#include <hikari/core/field.h>
#include <hikari/core/data_type.h>

auto hikari::core::Field::operator[](size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); } return Field(object->getChild(idx)); }

auto hikari::core::Field::operator[](size_t idx) -> FieldRef { auto object = getObject(); return FieldRef(object, idx); }

auto hikari::core::Field::clone() const -> Field
{
  auto object = getObject();
  if (!object) { return Field(nullptr); }
  return Field(object->clone());
}

auto hikari::core::Field::getSize() const -> size_t { return getChildCount(); }

void hikari::core::Field::setSize(size_t count) { setChildCount(count); }

auto hikari::core::Field::getKeys() const -> std::vector<Str> { auto object = getObject(); if (object) { return object->getKeys(); } else { return{}; } }
void hikari::core::Field::setName(const Str& name) { auto object = getObject(); if (object) { object->setName(name); } }

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

auto hikari::core::FieldRef::operator[](size_t idx) const -> Field { auto object = getObject(); if (!object) { return Field(); } return Field(object->getChild(idx)); }
auto hikari::core::FieldRef::operator[](size_t idx) -> FieldRef { auto object = getObject(); return FieldRef(object, idx); }


auto hikari::core::FieldRef::getSize() const -> size_t { return getChildCount(); }

void hikari::core::FieldRef::setSize(size_t count) { setChildCount(count); }

auto hikari::core::FieldRef::getKeys() const -> std::vector<Str> { auto object = getObject(); if (object) { return object->getKeys(); } else { return{}; } }

void hikari::core::FieldRef::setName(const Str& name) { auto object = getObject(); if (object) { object->setName(name); } }

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
      auto children = value.getValue<Array<Field>>();
      for (auto& child : children) {
        m_children.push_back(child.getObject());
      }
    }
    else {
      m_property_block.setValue(key, value);
    }
  }
}

void hikari::core::FieldObject::getPropertyBlock(PropertyBlock& pb) const {
  pb = m_property_block;
  pb.popValue("children");
  auto children = Array<Field>();
  pb.setValue("children", children);
}

bool hikari::core::FieldObject::hasProperty(const Str& name) const {
  if (name == "children") { return true; }
  return m_property_block.hasValue(name);
}

bool hikari::core::FieldObject::setProperty(const Str& name, const Property& value) {
  if (name == "children") {
    size_t type_index;
    if (value.getTypeIndex() == PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value)
    {
      auto children = value.getValue<Array<Field>>();
      m_children = {};
      if (!children.empty()) {
        for (auto& child : children) {
          m_children.push_back(child.getObject());
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
    std::vector<Field> objects = {};
    for (auto& child : m_children) { objects.push_back(Field(child)); }
    value.setValue(objects);
    return true;
  }
  else {
    return m_property_block.getValue(name, value);
  }
}

auto hikari::core::FieldObject::clone() const -> std::shared_ptr<FieldObject>
{
  auto field = FieldObject::create(m_name );
  field->setPropertyBlock(m_property_block);
  for (auto& child : m_children) {
    field->addChild(child->clone());
  }
  return field;
}

auto hikari::core::FieldObject::getKeys() const -> Array<Str>
{
  return m_property_block.getKeys();
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

hikari::core::FieldSerializer::~FieldSerializer() noexcept
{
}

auto hikari::core::FieldSerializer::getTypeString() const noexcept -> Str 
{
  return Field::type::TypeString();
}

auto hikari::core::FieldSerializer::eval(const std::shared_ptr<Object>& object) const -> Json 
{
  if (!object) { return Json(); }
  auto field = Field(std::static_pointer_cast<FieldObject>(object));
  Json json = {};
  json["type"] = "Field";
  json["name"] = object->getName();
  json["properties"] = {};
  auto children = field.getChildren();
  json["properties"]["children"] = Array<Json>();
  for (auto& child : children) {
    json["properties"]["children"].push_back(eval(child.getObject()));
  }
  for (auto& key : field.getKeys()) {
    if (key != "children") {
      json["properties"][key] = serialize(field.getValue(key));
    }
  }
  return json;
}

hikari::core::FieldDeserializer::~FieldDeserializer() noexcept
{
}

auto hikari::core::FieldDeserializer::getTypeString() const noexcept -> Str 
{
  return FieldObject::TypeString();
}

auto hikari::core::FieldDeserializer::eval(const Json& json) const -> std::shared_ptr<Object> 
{
  auto type = json.find("type");
  if (type == json.end()) { return nullptr; }
  if (!type.value().is_string()) { return nullptr; }
  auto str_type = type.value().get<Str>();
  if (str_type != "Field") { return nullptr; }
  auto name = json.find("name");
  if (name == json.end()) { return nullptr; }
  if (!name.value().is_string()) { return nullptr; }
  auto str_name = name.value().get<Str>();
  auto prop = json.find("properties");
  if (prop == json.end()) { return nullptr; }
  if (!prop.value().is_object()) { return nullptr; }
  auto field = Field(str_name);
  {
    auto items = prop.value().items();
    for (auto& item : items) {
      if (item.key() == "children") { continue; }
      auto prop = deserialize<Property>(item.value());
      field.setValue(item.key(), prop);
    }
  }

  auto children = prop.value().find("children");
  if (children == prop.value().end()) { return nullptr; }
  if (!children.value().is_array()) { return nullptr; }
  auto val_children = children.value().get<Array<Json>>();
  for (auto& child : val_children) {
    auto child_node = eval(child);
    if (!child_node) { return nullptr; }
    field.addChild(Field(std::static_pointer_cast<FieldObject>(child_node)));
  }
  return field.getObject();
}
