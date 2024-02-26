#include <hikari/window/system.h>
#include <hikari/window/window.h>
auto hikari::WindowSystem::getInstance() -> WindowSystem&
{
  // TODO: return ステートメントをここに挿入します
  static WindowSystem system;
  return system;
}

bool hikari::WindowSystem::isInitialized()
{
  return m_is_initialized;
}

bool hikari::WindowSystem::initialize()
{
  if (m_is_initialized) { return true; }
  return m_is_initialized = glfwInit();
}

void hikari::WindowSystem::terminate()
{
  if (!m_is_initialized) { return; }
  {
    glfwTerminate();
    m_is_initialized = false;
  }
}

