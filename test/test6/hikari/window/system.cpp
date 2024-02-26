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

auto hikari::WindowSystem::createWindow(const std::string& title, U32 width, U32 height, I32 pos_x, I32 pos_y, GraphicsAPIType api_type, Bool is_floating, Bool is_resizable, Bool is_visible, Bool is_fullscreen) -> Window*
{
  return new Window(title,width,height,pos_x,pos_y,api_type,is_floating,is_resizable,is_visible,is_fullscreen);
}

void hikari::WindowSystem::destroyWindow(Window* window)
{
  if (!window) { return; }
  delete window;
}
