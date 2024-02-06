#include <hikari/core/data_type.h>
#include <hikari/core/tuple.h>
#include <hikari/core/object.h>
#include <hikari/core/json.h>
#include <hikari/core/field.h>
#include <hikari/core/node.h>
#include <iostream>
#include <vector>
#include <optional>
#include <memory>
#include <utility>
#include <variant>
#include <unordered_map>
#include <string>
using namespace hikari;
struct NodeComponentSampleObject : public hikari::NodeComponentObject {
  using BaseType = hikari::NodeComponentObject;
  static inline Bool Convertible(const Str& str) noexcept {
    if (BaseType::Convertible(str)) { return true; }
    if (str == TypeString()) { return true; }
    return false;
  }
  static inline constexpr auto TypeString() -> const char* { return "NodeComponent"; }
  static auto create(const std::shared_ptr<NodeObject>& node) -> std::shared_ptr< NodeComponentSampleObject> {
    return std::shared_ptr<NodeComponentSampleObject>(new NodeComponentSampleObject(node));
  }
  NodeComponentSampleObject(const std::shared_ptr<NodeObject>& node) :NodeComponentObject(node) {}
  virtual ~NodeComponentSampleObject() noexcept {}

  // NodeComponentObject を介して継承されました
  auto getJSONString() const -> Str override
  {
    return "null";
  }
  auto getPropertyNames() const -> std::vector<Str> override
  {
    return std::vector<Str>();
  }
  void setPropertyBlock(const PropertyBlock& pb) override
  {
  }
  void getPropertyBlock(PropertyBlock& pb) const override
  {
  }
  Bool hasProperty(const Str& name) const override
  {
    return Bool();
  }
  Bool setProperty(const Str& name, const Property& value) override
  {
    return Bool();
  }
  Bool getProperty(const Str& name, Property& value) const override
  {
    return Bool();
  }
  Bool isConvertible(const Str& type_name) const override
  {
    return Convertible(type_name);
  }

  // NodeComponentObject を介して継承されました
  auto getTypeString() const -> Str override
  {
    return TypeString();
  }
};
struct NodeComponentSample {
  using TypesTuple = PropertyTypes;
public:
  using ObjectType = NodeComponentSampleObject;
  template<typename T>
  using Traits = in_tuple<T, TypesTuple>;
  using PropertyRef = ObjectPropertyRef;

  NodeComponentSample() noexcept :m_object{} {}
  NodeComponentSample(nullptr_t) noexcept :m_object{ } {}
  NodeComponentSample(const NodeComponentSample&) = default;
  NodeComponentSample& operator=(const NodeComponentSample&) = default;
  NodeComponentSample(const std::shared_ptr<NodeComponentSampleObject>& object) :m_object{ object } {}
  NodeComponentSample& operator=(const std::shared_ptr<NodeComponentSampleObject>& obj) { m_object = obj; return *this; }
  virtual ~NodeComponentSample() noexcept {}

  template<size_t N>
  auto operator[](const char(&name)[N])->PropertyRef { return operator[](Str(name)); }
  template<size_t N>
  auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
  auto operator[](const Str& name)->PropertyRef { return PropertyRef(getObject(),name); }
  auto operator[](const Str& name) const->Property { return getValue(name); }

  Bool operator!() const noexcept { return !getObject(); }
  operator Bool () const noexcept { return getObject() != nullptr; }

  void setPropertyBlock(const PropertyBlock& pb) { auto object = getObject(); if (!object) { return; } return object->setPropertyBlock(pb); }
  void getPropertyBlock(PropertyBlock& pb) const { auto object = getObject(); if (!object) { return; } return object->getPropertyBlock(pb); }

  auto getJSONString() const->std::string { auto object = getObject(); if (!object) { return ""; } return object->getJSONString(); }

  auto getName() const->Str { auto object = getObject(); if (!object) { return ""; } return object->getName(); }

  auto getObject() const -> std::shared_ptr<NodeComponentSampleObject> { return m_object.lock(); }
  auto getTypeString() const->Str { auto object = getObject(); if (!object) { return ""; } return object->getTypeString(); }

  Bool setValue(const Str& name, const Property& prop) { auto object = getObject(); if (!object) { return false; } return object->setProperty(name, prop); }
  Bool getValue(const Str& name, Property& prop) const { auto object = getObject(); if (!object) { return false; } return object->getProperty(name, prop); }
  auto getValue(const Str& name) const -> Property { auto object = getObject(); if (!object) { return Property(); }  Property prop; object->getProperty(name, prop); return prop; }
  Bool hasValue(const Str& name) const { auto object = getObject(); if (!object) { return false; } return object->hasProperty(name); }

  template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
  void setValue(const Str& name, T value) noexcept { return setValue(name, Property(value)); }
  template <size_t N>
  void setValue(const Str& name, const char(&value)[N]) noexcept { setValue(name, Str(value)); }

  Bool isConvertible(const Str& type_name) const {
    auto object = getObject(); if (!object) { return false; }
    return object->isConvertible(type_name);
  }
private:
  std::weak_ptr<NodeComponentSampleObject> m_object;
};
int main() {
  auto node  = hikari::Node(""   , hikari::Transform(hikari::Vec3(0.0f)));
  node.setChildCount(3);
  node[0]    = hikari::Node("0"  , hikari::Transform(hikari::Vec3(0.0f, 0.0f, 0.0f)));
  node[1]    = hikari::Node("1"  , hikari::Transform(hikari::Vec3(1.0f, 0.0f, 0.0f)));
  node[2]    = hikari::Node("2"  , hikari::Transform(hikari::Vec3(2.0f, 0.0f, 0.0f)));
  node[0].setChildCount(3);
  node[0][0] = hikari::Node("0-0", hikari::Transform(hikari::Vec3(0.0f, 0.0f, 0.0f)));
  node[0][1] = hikari::Node("0-1", hikari::Transform(hikari::Vec3(0.0f, 1.0f, 0.0f)));
  node[0][2] = hikari::Node("0-2", hikari::Transform(hikari::Vec3(0.0f, 2.0f, 0.0f)));
  node[1].setChildCount(3);
  node[1][0] = hikari::Node("1-0", hikari::Transform(hikari::Vec3(0.0f, 0.0f, 0.0f)));
  node[1][1] = hikari::Node("1-1", hikari::Transform(hikari::Vec3(0.0f, 1.0f, 0.0f)));
  node[1][2] = hikari::Node("1-2", hikari::Transform(hikari::Vec3(0.0f, 2.0f, 0.0f)));
  node[2].setChildCount(3);
  node[2][0] = hikari::Node("2-0", hikari::Transform(hikari::Vec3(0.0f, 0.0f, 0.0f)));
  node[2][1] = hikari::Node("2-1", hikari::Transform(hikari::Vec3(0.0f, 1.0f, 0.0f)));
  node[2][2] = hikari::Node("2-2", hikari::Transform(hikari::Vec3(0.0f, 2.0f, 0.0f)));
  auto sample = node[2][2].addComponent<NodeComponentSample>();
  //std::cout << node.getJSONString() << std::endl;
  hikari::ObjectWrapper opb = node;
  auto nodes = opb["children"].getValue<std::vector<std::shared_ptr<Object>>>();

}
