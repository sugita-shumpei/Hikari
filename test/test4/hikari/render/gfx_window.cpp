#include <hikari/render/gfx_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
auto hikari::render::GFXWindowManager::getInstance() -> GFXWindowManager& {
  static GFXWindowManager manager;
  return manager;
}

hikari::render::GFXWindowManager::~GFXWindowManager() noexcept { }

//void hikari::render::GFXWindowManager::addWindow(const GFXWindow& window) {
//  auto window_object = window.getObject();
//  if (!window_object) { return; }
//  if (m_windows.find(window_object.get()) == m_windows.end()) {
//    m_windows.insert({ window_object.get() ,window_object });
//  }
//}
//
//void hikari::render::GFXWindowManager::popWindow(const GFXWindow& window) {
//  auto window_object = window.getObject();
//  if (!window_object) { return; }
//  if (m_windows.find(window_object.get()) != m_windows.end()) {
//    m_windows.erase(window_object.get());
//  }
//}

void hikari::render::GFXWindowManager::update() {
  auto& manager = platforms::glfw::WindowManager::getInstance();
  manager.pollEvents();
}


hikari::render::GFXWindowManager::GFXWindowManager() noexcept  {

}
