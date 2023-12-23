#include <hikari/core/light.h>
#include <hikari/core/node.h>

hikari::Light::~Light() noexcept
{
}

auto hikari::Light::getNode() -> std::shared_ptr<Node> { return m_node.lock(); }

hikari::Light::Light() :m_node{}
{
}

void hikari::Light::onAttach(const std::shared_ptr<Node>& node)
{
  m_node = node;
}

void hikari::Light::onDetach()
{
  m_node = {};
}
