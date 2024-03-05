#include <hikari/core/window/renderer.h>
#include <hikari/core/window/window.h>
#include "impl_glfw_context.h"

hikari::core::WindowRenderer::WindowRenderer(Window* window):
  m_window{window},
  m_surface_size{window->getSurfaceSize()}
{
}

hikari::core::WindowRenderer::~WindowRenderer() noexcept
{
}

void hikari::core::WindowRenderer::update()
{
}

auto hikari::core::WindowRenderer::getSurfaceSize() const noexcept -> WindowExtent2D
{
  return WindowExtent2D();
}

void hikari::core::WindowRenderer::onSync()
{
  m_surface_size  = getSurfaceSize();// 同期中なのでスレッドセーフ
  m_on_resize     = false;// 同期中なのでスレッドセーフ
  m_on_fullscreen = false;// 同期中なのでスレッドセーフ
  m_on_windowed   = false;// 同期中なのでスレッドセーフ
}

auto hikari::core::WindowRenderer::vkCreateSurface(void* instance)->void*
{
  assert(m_window->isGraphicsVulkan());
  VkSurfaceKHR surface = nullptr;
  if (glfwCreateWindowSurface((VkInstance)instance, (GLFWwindow*)m_window->getNativeHandle(), nullptr, &surface) == VK_SUCCESS) {
    return surface;
  }
  else {
    return nullptr;
  }
}

auto hikari::core::WindowRenderer::getRenderThread() noexcept -> BS::thread_pool*
{
  return m_window->getRenderThread();
}

void hikari::core::WindowRenderer::glSwapBuffers()
{
  assert(m_window->isGraphicsOpenGL());
  glfwSwapBuffers((GLFWwindow*)m_window->getNativeHandle());
}

auto hikari::core::WindowRenderer::glCreateSharingContext()const -> void*
{
  assert(m_window->isGraphicsOpenGL());
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(1, 1, "offscreen", nullptr, (GLFWwindow*)m_window->getNativeHandle());
  glfwDefaultWindowHints();
  return window;
}

void hikari::core::WindowRenderer::glDestroySharingContext(void* handle)const
{
  assert(m_window->isGraphicsOpenGL());
  glfwDestroyWindow((GLFWwindow*)handle);
}

void hikari::core::WindowRenderer::glSetCurrentContext(void* handle)const
{
  assert(m_window->isGraphicsOpenGL());
  glfwMakeContextCurrent((GLFWwindow*)handle);
}

void hikari::core::WindowRenderer::glSwapInterval(int interval)const
{
  assert (m_window->isGraphicsOpenGL());
  glfwSwapInterval(interval);
}

void hikari::core::WindowRenderer::glSetCurrentContext()const
{
  assert(m_window->isGraphicsOpenGL());
  glSetCurrentContext(m_window->getNativeHandle());
}

auto hikari::core::WindowRenderer::glGetProcAddress()->Window_PFN_glProcAddr(*)(const char*)
{
  return glfwGetProcAddress;
}
