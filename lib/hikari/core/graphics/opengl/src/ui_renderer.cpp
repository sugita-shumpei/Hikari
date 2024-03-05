#include <hikari/core/graphics/opengl/ui_renderer.h>
#include <hikari/core/graphics/opengl/renderer.h>
#include <hikari/core/graphics/opengl/context.h>
#include <hikari/core/window/window.h>
#include <hikari/core/ui/manager.h>
hikari::core::GraphicsOpenGLUIRenderer::GraphicsOpenGLUIRenderer(Window* window) :UIRenderer(window) {
  if (!window) { throw std::runtime_error("Null Window Cannot Use For UI Render Creation!"); }
  if (!window->isGraphicsOpenGL()) { throw std::runtime_error("Not Compatible Graphics!"); }
}
bool hikari::core::GraphicsOpenGLUIRenderer::initialize()
{
  std::future<bool> future = this->executeInRenderThread([this](){
    auto renderer = (GraphicsOpenGLRenderer*)this->getRenderer();
    renderer->setContextCurrent();
    if (!ImGui_ImplOpenGL3_Init("#version 460 core")) { return false; }
    if (!ImGui_ImplOpenGL3_CreateDeviceObjects()) { return false; }
    return true;
  });
  if (!future.get()) {
    return false;
  }

  return true;
}
void hikari::core::GraphicsOpenGLUIRenderer::terminate()
{
  std::future<void> future = this->executeInRenderThread([this](){
    auto renderer = (GraphicsOpenGLRenderer*)this->getRenderer();
    renderer->setContextCurrent();
    return ImGui_ImplOpenGL3_Shutdown();
  });
  future.wait();
}

void hikari::core::GraphicsOpenGLUIRenderer::update()
{
  // 新しいフレームの更新
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplOpenGL3_RenderDrawData(&getDrawData());
}
