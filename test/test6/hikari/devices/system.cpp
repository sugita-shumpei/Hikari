#include <hikari/devices/system.h>
#include <hikari/devices/window.h>
#include <hikari/console/log_system.h>
#include "system.h"
#include "system.h"
#include "system.h"
#include "system.h"
#include "system.h"
#include "system.h"
#include "system.h"
#include "system.h"
using namespace hikari;
bool hikari::DeviceSystemImpl::initialize()
{
  if (m_is_initialized) { return true; }
  m_is_initialized = glfwInit();
  if (m_is_initialized) {
    HK_CORE_INFO("[SUCC] Init DeviceSystem");
  }
  else {
    HK_CORE_FATAL("[FAIL] Init DeviceSystem");
  }
  return m_is_initialized;
}

void hikari::DeviceSystemImpl::terminate() noexcept
{
  if (!m_is_initialized) { return; }
  for (auto& window : m_windows) {
    delete window;
  }
  m_windows.clear();
  HK_CORE_INFO("[INFO] Free DeviceSystem");
  glfwTerminate();
  m_is_initialized = false;
}

hikari::DeviceSystemImpl::DeviceSystemImpl() noexcept
{
}

hikari::DeviceSystemImpl::~DeviceSystemImpl() noexcept
{
}

bool hikari::DeviceSystemImpl::isInitialized() const noexcept
{
  return m_is_initialized;
}
// Window 
auto hikari::DeviceSystemImpl::createWindow(const WindowCreateDesc& desc) -> Window* {
  auto window = new Window(desc);
  m_windows.push_back(window);
  return window;
}

void hikari::DeviceSystemImpl::destroyWindow(Window* window)
{
  if (!window) { return; }
  auto iter = std::ranges::find(m_windows, window);
  if (iter != std::end(m_windows)){
    delete window;
    m_windows.erase(iter);
  }
}

void hikari::DeviceSystemImpl::update()
{
    glfwPollEvents();
    for (auto& window : m_windows) {
      window->update();
    }
}
