#include <hikari/graphics/system.h>
#include <cstdint>
#include <hikari/graphics/opengl/instance.h>

auto hikari::GraphicsSystem::getInstance() -> GraphicsSystem&
{
  // TODO: return ステートメントをここに挿入します
  static GraphicsSystem system;
  return system;
}

bool hikari::GraphicsSystem::initGraphics(GraphicsAPIType api)
{
  if (m_instances[(size_t)api]) {
    return true;
  }
  if (api == GraphicsAPIType::eOpenGL)
  {
    auto instance = new GraphicsOpenGLInstance();
    if (!instance->initialize()) {
      instance->terminate();
      delete instance;
      return false;
    }
    m_instances[(size_t)GraphicsAPIType::eOpenGL] = instance;
    return true;
  }
  return false;
}

void hikari::GraphicsSystem::freeGraphics(GraphicsAPIType api)
{
  if (!m_instances[(size_t)api]) {
    return;
  }
  m_instances[(size_t)api]->terminate();
  m_instances[(size_t)api] = nullptr;
}

auto hikari::GraphicsSystem::getGraphics(GraphicsAPIType api) -> GraphicsInstance*
{
  return m_instances[(size_t)api];
}

