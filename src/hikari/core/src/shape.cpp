#include <hikari/core/shape.h>
#include <hikari/core/node.h>

hikari::Shape::~Shape() noexcept
{
}

void hikari::Shape::setFlipNormals(Bool flip_normals)
{
  m_flip_normals = true;
}

auto hikari::Shape::getFlipNormals() const -> Bool
{
    return m_flip_normals;
}

auto hikari::Shape::getNode() -> std::shared_ptr<Node> { return m_node.lock(); }

void hikari::Shape::setMaterial(const std::shared_ptr<Material>& material)
{
  m_material = material;
}

auto hikari::Shape::getMaterial() const -> std::shared_ptr<Material>
{
  return m_material;
}

hikari::Shape::Shape() :m_node{}, m_material{nullptr}
{
}

void hikari::Shape::onAttach(const std::shared_ptr<Node>& node)
{
  m_node = node;
}

void hikari::Shape::onDetach()
{
  m_node = {};
}
