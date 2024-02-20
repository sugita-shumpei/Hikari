#include <hikari/platform/opengl/render/gfx_opengl_instance.h>
#include <glad/glad.h>
#include <hikari/platform/opengl/render/gfx_opengl_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
auto hikari::platforms::opengl::render::GFXOpenGLInstanceObject::create() -> std::shared_ptr<GFXOpenGLInstanceObject> {
  return std::shared_ptr<GFXOpenGLInstanceObject>(new GFXOpenGLInstanceObject());
}

hikari::platforms::opengl::render::GFXOpenGLInstanceObject::GFXOpenGLInstanceObject() noexcept : hikari::render::GFXInstanceObject() {
  auto& manager = glfw::WindowManager::getInstance();
  m_main_window = manager.createOpenGLWindow(1, 1, -1,-1,"main context", false, false, nullptr);
  manager.setCurrentContext(m_main_window);
  if (gladLoadGLLoader((GLADloadproc)manager.GetProcAddress)) {}
  manager.setCurrentContext(nullptr);
}

hikari::platforms::opengl::render::GFXOpenGLInstanceObject::~GFXOpenGLInstanceObject() noexcept {
  auto& manager = glfw::WindowManager::getInstance();
  if (m_main_window) {
    manager.destroyWindow(m_main_window);
    m_main_window = nullptr;
  }
}

auto hikari::platforms::opengl::render::GFXOpenGLInstanceObject::createOpenGLWindow(const GFXWindowDesc& desc) -> std::shared_ptr<GFXOpenGLWindowObject> {
  return GFXOpenGLWindowObject::create(shared_from_this(), desc);
}

auto hikari::platforms::opengl::render::GFXOpenGLInstanceObject::createWindow(const GFXWindowDesc& desc) -> std::shared_ptr<hikari::render::GFXWindowObject>  {
  return std::static_pointer_cast<hikari::render::GFXWindowObject>(createOpenGLWindow(desc));
}

auto hikari::platforms::opengl::render::GFXOpenGLInstanceObject::getMainContext() const -> void* { return m_main_window; }

auto hikari::platforms::opengl::render::GFXOpenGLInstance::createWindow(const GFXWindowDesc& desc) -> GFXOpenGLWindow {
  auto obj = getObject();
  if (!obj) { return GFXOpenGLWindow(nullptr); }
  return GFXOpenGLWindow(obj->createOpenGLWindow(desc));
}
