#include <hikari/core/scene.h>
#include <hikari/core/node.h>
auto hikari::Scene::create(const String& name) -> std::shared_ptr<Scene> {
  auto res = std::shared_ptr<Scene>(new Scene(name));
  res->m_root_node->onAttachScene(res);
  return res;
}
hikari::Scene::~Scene() {

}
hikari::Scene::Scene(const String& name) :m_root_node{Node::create(name,true)} {

}
void hikari::Scene::addChild(const std::shared_ptr<Node>& node)
{
  node->setParent(m_root_node);
}

auto hikari::Scene::getChildren() -> std::vector<std::shared_ptr<Node>>
{
  return m_root_node->getChildren();
}

auto hikari::Scene::getChildCount() const -> U32 { return m_root_node->getChildCount(); }

auto hikari::Scene::getChild(U32 idx) -> std::shared_ptr<Node> { return m_root_node->getChild(idx); }

void hikari::Scene::setName(const String& name)
{
  m_root_node->setName(name);
}

auto hikari::Scene::getName() const->String {
  return m_root_node->getName();
}

auto hikari::Scene::getNodesInHierarchy() -> std::vector<std::shared_ptr<Node>>
{
  return m_root_node->getNodesInHierarchy();
}

auto hikari::Scene::getCameras() -> std::vector<std::shared_ptr<Camera>>
{
  return m_root_node->getCameras();
}

auto hikari::Scene::getLights()  -> std::vector<std::shared_ptr<Light>>
{
  return m_root_node->getLights();
}

auto hikari::Scene::getShapes()  -> std::vector<std::shared_ptr<Shape>>
{
  return m_root_node->getShapes();
}

