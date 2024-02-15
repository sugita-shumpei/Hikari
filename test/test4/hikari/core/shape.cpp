#include <hikari/core/shape.h>

auto hikari::core::ShapeFilterObject::getNode() const -> std::shared_ptr<NodeObject> 
{
  return m_node.lock();
}

void hikari::core::ShapeFilterObject::setShape(const std::shared_ptr<ShapeObject>& shape)
{
  m_shape = shape;
}

auto hikari::core::ShapeFilterObject::getShape() const -> std::shared_ptr<ShapeObject>
{
  return m_shape;
}

hikari::Str hikari::core::ShapeFilterObject::getTypeString() const noexcept
{
  return TypeString();
}

bool hikari::core::ShapeFilterObject::isConvertible(const Str& type) const noexcept
{
  return Convertible(type);
}

auto hikari::core::ShapeFilterObject::getPropertyNames() const -> std::vector<Str>
{
  return {"node","shape"};
}

void hikari::core::ShapeFilterObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const
{
  pb.setValue("node",   Node(getNode()));
  pb.setValue("shape",Shape(getShape()));
}

void hikari::core::ShapeFilterObject::setPropertyBlock(const PropertyBlockBase<Object>& pb)
{
  auto shape = pb.getValue<Shape>("shape");
  if (shape) {
    setShape(shape.getObject());
  }
}

bool hikari::core::ShapeFilterObject::hasProperty(const Str& name) const
{
  if (name == "shape") { return true; }
  if (name == "node" ) { return true; }
  return false;
}

bool hikari::core::ShapeFilterObject::getProperty(const Str& name, PropertyBase<Object>& prop) const
{
  if (name == "shape") { prop.setValue(Shape(getShape())); return true; }
  if (name == "node")  { prop.setValue(Node(getNode())); return true; }
  return false;
}

bool hikari::core::ShapeFilterObject::setProperty(const Str& name, const PropertyBase<Object>& prop)
{
  if (name == "shape") {
    auto shape = prop.getValue<Shape>();
    if (shape) { setShape(shape.getObject()); }
    return true;
  }
  return false;
}

hikari::Str hikari::core::ShapeRenderObject::getTypeString() const noexcept
{
  return TypeString();
}

bool hikari::core::ShapeRenderObject::isConvertible(const Str& type) const noexcept
{
  return Convertible(type);
}

auto hikari::core::ShapeRenderObject::getPropertyNames() const -> std::vector<Str> 
{
  return { "node" };
}

void hikari::core::ShapeRenderObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const
{
  pb.setValue("node" , Node(getNode()));
}

void hikari::core::ShapeRenderObject::setPropertyBlock(const PropertyBlockBase<Object>& pb)
{
}

bool hikari::core::ShapeRenderObject::hasProperty(const Str& name) const
{
  if (name == "shape") { return true; }
  if (name == "node") { return true; }
  return false;
}

bool hikari::core::ShapeRenderObject::getProperty(const Str& name, PropertyBase<Object>& prop) const
{
  if (name == "node") { prop.setValue<Node>(getNode()); return true; }
  return false;
}

bool hikari::core::ShapeRenderObject::setProperty(const Str& name, const PropertyBase<Object>& prop)
{
  return false;
}

auto hikari::core::ShapeRenderObject::getNode() const -> std::shared_ptr<NodeObject> 
{
  return m_node.lock();
}

auto hikari::core::ShapeFilterDeserializer::getTypeString() const noexcept -> Str
{
  return ShapeFilterObject::TypeString();
}

auto hikari::core::ShapeFilterDeserializer::eval(const std::shared_ptr<NodeObject>& node, const Json& json) const -> std::shared_ptr<NodeComponentObject>
{
  if (!node) { return nullptr; }
  auto prop_iter = json.find("properties");
  if (prop_iter == json.end()) { return nullptr; }
  if (!prop_iter.value().is_object()) { return nullptr; }
  auto shape_iter = prop_iter.value().find("shape");
  if (shape_iter == prop_iter.value().end()) { return nullptr; }
  auto shape = deserialize<Shape>(shape_iter.value());
  if (!shape) { return nullptr; }
  auto comp = node->addComponent<ShapeFilterObject>();
  comp->setShape(shape.getObject());
  return comp;
}

auto hikari::core::ShapeFilterSerializer::getTypeString() const noexcept -> Str
{
  return ShapeFilterObject::TypeString();
}

auto hikari::core::ShapeFilterSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto shape_filter = ObjectUtils::convert<ShapeFilterObject>(object);
  if (!shape_filter) { return Json(); }
  Json json;
  json["type"] = "ShapeFilter";
  json["properties"] = {};
  auto shape = Shape(shape_filter->getShape());
  json["properties"]["shape"] = serialize(shape);
  return json;
}

auto hikari::core::ShapeRenderDeserializer::getTypeString() const noexcept -> Str 
{
  return ShapeRenderObject::TypeString();
}

auto hikari::core::ShapeRenderDeserializer::eval(const std::shared_ptr<NodeObject>& node, const Json& json) const -> std::shared_ptr<NodeComponentObject>
{
  return nullptr;
}

auto hikari::core::ShapeRenderSerializer::getTypeString() const noexcept -> Str
{
  return ShapeRenderObject::TypeString();
}

auto hikari::core::ShapeRenderSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto shape_render = ObjectUtils::convert<ShapeRenderObject>(object);
  if (!shape_render) { return Json(); }
  Json json;
  json["type"] = "ShapeRender";
  json["properties"] = {};
  return json;
}
