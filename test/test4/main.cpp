#include <hikari/core/data_type.h>
#include <hikari/core/tuple.h>
#include <hikari/core/object.h>
#include <hikari/core/field.h>
#include <hikari/core/node.h>
#include <glm/gtx/string_cast.hpp>
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
  using base_type = hikari::NodeComponentObject;
  static inline Bool Convertible(const Str& str) noexcept {
    if (base_type::Convertible(str)) { return true; }
    if (str == TypeString()) { return true; }
    return false;
  }
  static inline constexpr auto TypeString()noexcept -> const char* { return "NodeComponentSample"; }
  static auto create(const std::shared_ptr<NodeObject>& node) -> std::shared_ptr< NodeComponentSampleObject> {
    return std::shared_ptr<NodeComponentSampleObject>(new NodeComponentSampleObject(node));
  }
  NodeComponentSampleObject(const std::shared_ptr<NodeObject>& node) :NodeComponentObject(), m_node{ node } {}
  virtual ~NodeComponentSampleObject() noexcept {}

  // NodeComponentObject を介して継承されました
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
  Bool isConvertible(const Str& type_name) const noexcept override
  {
    return Convertible(type_name);
  }

  // NodeComponentObject を介して継承されました
  auto getTypeString() const noexcept-> Str override
  {
    return TypeString();
  }
  // NodeComponentObject を介して継承されました
  auto getNode() const -> std::shared_ptr<NodeObject> override
  {
    return std::shared_ptr<NodeObject>(m_node.lock());
  }
private:
  std::weak_ptr<NodeObject> m_node;

};
struct NodeComponentSample : protected ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef,NodeComponentSampleObject>{
  using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef, NodeComponentSampleObject>;
  using type = typename impl_type::type;

  NodeComponentSample() noexcept :impl_type() {}
  NodeComponentSample(nullptr_t) noexcept :impl_type() {}
  NodeComponentSample(const NodeComponentSample&) = default;
  NodeComponentSample& operator=(const NodeComponentSample&) = default;
  NodeComponentSample(const std::shared_ptr<NodeComponentSampleObject>& object) :impl_type(object) {}
  NodeComponentSample& operator=(const std::shared_ptr<NodeComponentSampleObject>& obj) { setObject(obj); return *this; }
  ~NodeComponentSample() noexcept {}

  using impl_type::operator!;
  using impl_type::operator bool;
  using impl_type::operator[];
  using impl_type::isConvertible;
  using impl_type::getName;
  using impl_type::getKeys;
  using impl_type::getObject;
  using impl_type::getPropertyBlock;
  using impl_type::setPropertyBlock;
  using impl_type::getValue;
  using impl_type::hasValue;
  using impl_type::setValue;
};
int main() {
  ObjectSerializeManager::getInstance().add(std::make_shared<NodeSerializer>());

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
  hikari::ObjectWrapper opb = node;
  auto nodes = opb["children"].getValue<std::vector<Node>>();

  auto transforms = node.getComponentsInChildren<NodeTransform>();
  for (auto& transform : transforms) {
    std::cout << transform.getName() << ": ";
    std::cout << glm::to_string(*transform.getGlobalPosition()) << std::endl;
  }

  auto json = ObjectSerializeManager::getInstance().serialize(node.getObject());
  std::cerr << json.dump() << std::endl;
}
