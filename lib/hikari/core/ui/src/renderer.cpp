#include <hikari/core/ui/renderer.h>
#include <hikari/core/ui/manager.h>
#include <hikari/core/ui/common.h>
#include <hikari/core/window/window.h>
#include <hikari/core/window/renderer.h>
hikari::core::UIRenderer::UIRenderer(Window* window)
  :m_window{ window },m_manager{nullptr}, m_draw_data{}
{
}

hikari::core::UIRenderer::~UIRenderer() noexcept
{
  imgui_utils::free_draw_data(m_draw_data);
}

bool hikari::core::UIRenderer::initialize() { return true; }

// 初期化

void hikari::core::UIRenderer::terminate() {

}

void hikari::core::UIRenderer::update()
{
}

auto hikari::core::UIRenderer::getWindow() -> Window*
{
  return m_window;
}

auto hikari::core::UIRenderer::getRenderer() -> WindowRenderer*
{
  auto window = getWindow();
  if (!window) { return nullptr; }
  return window->getRenderer();
}

auto hikari::core::UIRenderer::getRenderThread() -> BS::thread_pool*
{
  auto renderer = getRenderer();
  if (!renderer) { return nullptr; }
  return renderer->getRenderThread();
}

void hikari::core::UIRenderer::setManager(UIManager* manager)
{
  m_manager = manager;
}

auto hikari::core::UIRenderer::getManager() -> UIManager*
{
  return m_manager;
}

auto hikari::core::UIRenderer::getDrawData() -> ImDrawData&
{
  // TODO: return ステートメントをここに挿入します
  return m_draw_data;
}
