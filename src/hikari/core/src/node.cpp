#include <hikari/core/node.h>
#include <hikari/core/camera.h>
#include <hikari/core/light.h>
#include <hikari/core/shape.h>
#include <hikari/core/material.h>
#include <stack>

hikari::Node::Node(const String& name ,Bool isRootOnly)
  : m_name{name},
    m_scene{},
    m_parent{},
    m_children{},
    m_is_root_only{isRootOnly},
    m_camera{},
    m_light{},
    m_shape{},
    m_local_transform{},
    m_parent_transform{}
{
}

hikari::Node::~Node(){}

void hikari::Node::addChild(const std::shared_ptr<Node>& node)
{
  m_children.push_back(node);
}
void hikari::Node::popChild(const std::shared_ptr<Node>& node)
{
  auto iter = std::ranges::find(m_children, node);
  if (iter != std::end(m_children)) {
      m_children.erase(iter);
  }
}
void hikari::Node::onAttachScene(const std::shared_ptr<Scene>& scene)
{
  m_scene = scene;
}
void hikari::Node::onDetachScene()
{
  m_scene = {};
}
void hikari::Node::updateTransform()
{
  auto parent = m_parent.lock();
  if (parent) {
    m_parent_transform = parent->getGlobalTransform();
  }
  else {
    m_parent_transform = m_local_transform;
  }
  for (auto& child : m_children) {
    child->updateTransform();
  }
}
auto hikari::Node::create(const String& name ,Bool isRootOnly) -> std::shared_ptr<Node>
{
    return std::shared_ptr<Node>(new Node(name,isRootOnly));
}
auto hikari::Node::getChild(U32 idx) -> std::shared_ptr<Node>
{
    if (m_children.size() > idx) { return m_children[idx];}
    else { return nullptr; }
}
auto hikari::Node::getScene() -> std::shared_ptr<Scene>
{
  return m_scene.lock();
}
void hikari::Node::setCamera(const std::shared_ptr<Camera>& camera)
{
  if (m_camera) {
    m_camera->onDetach();
  }
  m_camera = camera;
  if (m_camera) {
    m_camera->onAttach(shared_from_this());
  }
}
auto hikari::Node::getCamera() -> std::shared_ptr<Camera>
{
  return m_camera;
}
void hikari::Node::setLight(const std::shared_ptr<Light>& light)
{
  if (m_light) {
    m_light->onDetach();
  }
  m_light = light;
  if (m_light) {
    m_light->onAttach(shared_from_this());
  }
}
auto hikari::Node::getLight() -> std::shared_ptr<Light>
{
  return m_light;
}

auto hikari::Node::getNodesInHierarchy() -> std::vector<std::shared_ptr<Node>>
{
  auto nodes = std::vector<hikari::NodePtr>();
  std::stack<NodePtr> stack_nodes = {};
  {
    for (auto& child : m_children) {
      stack_nodes.push(child);
      nodes.push_back(child);
    }
  }
  while (!stack_nodes.empty())
  {
    auto child = stack_nodes.top();
    stack_nodes.pop();
    nodes.reserve(nodes.size() + child->getChildCount());
    for (U64 i = 0; i < child->getChildCount(); ++i) {
      auto ch = child->getChild(i);
      stack_nodes.push(ch);
      nodes.push_back(ch);
    }
  }
  return nodes;
}
auto hikari::Node::getCameras() -> std::vector<std::shared_ptr<Camera>>
{
  auto cameras = std::vector<hikari::CameraPtr>();
  {
    auto camera = getCamera();
    if (camera) {
      cameras.push_back(camera);
    }
  }
  std::stack<NodePtr> stack_nodes = {};
  {
    for (auto& child : m_children) {
      stack_nodes.push(child);
      auto camera = child->getCamera();
      if (camera) {
        cameras.push_back(camera);
      }
    }
  }
  while (!stack_nodes.empty())
  {
    auto child = stack_nodes.top();
    stack_nodes.pop();
    cameras.reserve(cameras.size() + child->getChildCount());
    for (U64 i = 0; i < child->getChildCount(); ++i) {
      auto ch = child->getChild(i);
      auto ch_camera = ch->getCamera();
      stack_nodes.push(ch);
      if (ch_camera) {
        cameras.push_back(ch_camera);
      }
    }
  }
  return cameras;
}
auto hikari::Node::getLights() -> std::vector<std::shared_ptr<Light>>
{
  auto lights = std::vector<hikari::LightPtr>();
  {
    auto light = getLight();
    if (light) {
      lights.push_back(light);
    }
  }
  std::stack<NodePtr> stack_nodes = {};
  {
    for (auto& child : m_children) {
      stack_nodes.push(child);

      auto light = child->getLight();
      if (light) {
        lights.push_back(light);
      }
    }
  }
  while (!stack_nodes.empty())
  {
    auto child = stack_nodes.top();
    stack_nodes.pop();
    lights.reserve(lights.size() + child->getChildCount());
    for (U64 i = 0; i < child->getChildCount(); ++i) {
      auto ch = child->getChild(i);
      auto ch_light = ch->getLight();
      stack_nodes.push(ch);
      if (ch_light) {
        lights.push_back(ch_light);
      }
    }
  }
  return lights;
}
auto hikari::Node::getShapes() -> std::vector<std::shared_ptr<Shape>>
{
  auto shapes = std::vector<hikari::ShapePtr>();
  {
    auto shape = getShape();
    if (shape) {
      shapes.push_back(shape);
    }
  }
  std::stack<NodePtr> stack_nodes = {};
  {
    for (auto& child : m_children) {
      stack_nodes.push(child);

      auto shape = child->getShape();
      if (shape) {
        shapes.push_back(shape);
      }
    }
  }
  while (!stack_nodes.empty())
  {
    auto child = stack_nodes.top();
    stack_nodes.pop();
    shapes.reserve(shapes.size() + child->getChildCount());
    for (U64 i = 0; i < child->getChildCount(); ++i) {
      auto ch = child->getChild(i);
      auto ch_shape = ch->getShape();
      stack_nodes.push(ch);
      if (ch_shape) {
        shapes.push_back(ch_shape);
      }
    }
  }
  return shapes;
}
void hikari::Node::setShape(const std::shared_ptr<Shape> &shape)
{
  if (m_shape) {
    m_shape->onDetach();
  }
  m_shape = shape;
  if (m_shape) {
    m_shape->onAttach(shared_from_this());
  }
}
auto hikari::Node::getShape()->std::shared_ptr<Shape>
{
    return m_shape;
}
auto hikari::Node::getLocalPosition() const -> std::optional<Vec3>
{
  return m_local_transform.getPosition();
}
auto hikari::Node::getLocalRotation() const -> std::optional<Quat>
{
  return m_local_transform.getRotation();
}
auto hikari::Node::getLocalScale() const -> std::optional<Vec3>
{
  return m_local_transform.getScale();
}
auto hikari::Node::getLocalTransform() const -> Transform
{
    return m_local_transform;
}
void hikari::Node::setGlobalTransform(const Transform& transform)
{
  setLocalTransform(m_parent_transform.inverse() * transform);
}
auto hikari::Node::getGlobalPosition() const -> std::optional<Vec3>
{
  auto global = getGlobalTransform();
  return global.getPosition();
}
auto hikari::Node::getGlobalRotation() const -> std::optional<Quat>
{
  auto global = getGlobalTransform();
  return global.getRotation();
}
auto hikari::Node::getGlobalScale() const -> std::optional<Vec3>
{
  auto global = getGlobalTransform();
  return global.getScale();
}
auto hikari::Node::getGlobalTransform() const -> Transform
{
  return m_parent_transform * m_local_transform;
}
auto hikari::Node::getParentTransform() const -> Transform
{
  return m_parent_transform;
}
void hikari::Node::setLocalTransform(const Transform &transform)
{
  m_local_transform = transform;
  for (auto& child : m_children) {
    child->updateTransform();
  }
}
auto hikari::Node::getParent() -> std::shared_ptr<Node>
{
  return m_parent.lock();
}
void hikari::Node::setParent(const std::shared_ptr<Node> &parent)
{
    if (m_is_root_only) { return; }
    auto old_par = m_parent.lock();
    if (old_par){
        old_par->popChild(shared_from_this());
    }
    m_parent = parent;
    if (parent){
        parent->addChild(shared_from_this());
        onAttachScene(parent->getScene());
    }
    else {
        onDetachScene();
    }
    updateTransform();
}

auto hikari::Node::getChildren() -> std::vector<std::shared_ptr<Node>>
{
    return m_children;
}
auto hikari::Node::getChildCount() const -> U32
{
    return m_children.size();
}
