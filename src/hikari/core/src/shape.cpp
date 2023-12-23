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

hikari::Shape::Shape() :m_node{}
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
