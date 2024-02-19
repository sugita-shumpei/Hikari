#include <hikari/platform/opengl/render/gfx_opengl_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
auto hikari::platforms::opengl::render::GFXOpenGLWindowObject::create(const std::shared_ptr<GFXOpenGLInstanceObject>& instance, const GFXWindowDesc& desc) -> std::shared_ptr<GFXOpenGLWindowObject> {
  int x = desc.x == UINT32_MAX ? -1 : desc.x;
  int y = desc.y == UINT32_MAX ? -1 : desc.y;
  auto& manager = glfw::WindowManager::getInstance();
  auto wnd = manager.createOpenGLWindow(desc.width, desc.height, x, y, desc.title, desc.is_visible, desc.is_resizable, desc.is_borderless, nullptr);
  if (!wnd) { return nullptr; }
  auto res = std::shared_ptr<GFXOpenGLWindowObject>(new GFXOpenGLWindowObject(instance,(GLFWwindow*)wnd, desc));
  if (res) { res->initEvents(); }
  return res;
}
hikari::platforms::opengl::render::GFXOpenGLWindowObject::~GFXOpenGLWindowObject() noexcept {}

hikari::platforms::opengl::render::GFXOpenGLWindowObject::GFXOpenGLWindowObject(const std::shared_ptr<GFXOpenGLInstanceObject>& instance,GLFWwindow* window, const GFXWindowDesc& desc) :
  hikari::platforms::glfw::render::GFXGLFWWindowObject(window,desc), m_instance{ instance }{
}
