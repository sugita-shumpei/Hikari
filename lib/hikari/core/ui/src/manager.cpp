#include <hikari/core/ui/manager.h>
#include <hikari/core/window/window.h>
#include <hikari/core/window/renderer.h>
#include <hikari/core/ui/renderer.h>
#include <hikari/core/ui/common.h>
hikari::core::UIManager::UIManager(UIRenderer* renderer):m_renderer{renderer}
{
}

bool hikari::core::UIManager::initialize()
{
  if (!initUI()) { return false; }
  if (!initRenderer()) { return false; }
  return true;
}

void hikari::core::UIManager::terminate()
{
  freeRenderer();
  freeUI();
}

void hikari::core::UIManager::update()
{
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  onUpdate();
  ImGui::Render();
  // 描画状態を送信
}

void hikari::core::UIManager::render()
{
  auto renderer = m_renderer.get();
  if (renderer) renderer->update();
}

void hikari::core::UIManager::onSync()
{
  auto draw_data = ImGui::GetDrawData();
  if (draw_data) {
    imgui_utils::clone_draw_data(*draw_data, m_renderer->getDrawData());
  }
}

auto hikari::core::UIManager::getWindow() -> Window*
{
  return m_renderer->getWindow();
}

bool hikari::core::UIManager::isHoveredMouse() const noexcept
{
  return ImGui::GetIO().WantCaptureMouse;
}

bool hikari::core::UIManager::initUI()
{
  IMGUI_CHECKVERSION();
  (void)ImGui::CreateContext();
  auto& io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  auto window = getWindow();
  if (window->isGraphicsOpenGL()) {
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)window->getNativeHandle(), true);
    auto renderer = window->getRenderer();
  }
  if (window->isGraphicsVulkan()) {
    ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)window->getNativeHandle(), true);
    auto renderer = window->getRenderer();
  }
  // Rendererのセットアップ
  m_is_init_ui = true;
  return true;
}

void hikari::core::UIManager::freeUI()
{
  if (!m_is_init_ui) { return; }
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  m_is_init_ui = false;
}

bool hikari::core::UIManager::initRenderer()
{
  if (m_is_init_renderer) { return false; }
  if (!m_renderer->initialize()) { return false; }
  m_renderer->setManager(this);
  m_is_init_renderer = true;
  return true;
}

void hikari::core::UIManager::freeRenderer()
{
  if (!m_is_init_renderer) { return; }
  m_renderer->setManager(nullptr);
  m_renderer->terminate();
  m_renderer.reset();
  m_is_init_renderer = false;
}

void hikari::core::UIManager::onUpdate()
{
  ImGui::ShowDemoWindow();
}
