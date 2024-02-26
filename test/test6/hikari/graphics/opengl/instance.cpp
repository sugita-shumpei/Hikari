#include <hikari/graphics/opengl/instance.h>

hikari::GraphicsOpenGLInstance::~GraphicsOpenGLInstance() noexcept
{
}

auto hikari::GraphicsOpenGLInstance::getHandle() const -> GLFWwindow*
{
  return m_window;
}

hikari::GraphicsOpenGLInstance::GraphicsOpenGLInstance() noexcept
{
}

bool hikari::GraphicsOpenGLInstance::initialize()
{
  if (m_is_initialized) { return true; }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* offscreen_ctx = glfwCreateWindow(1, 1, "offscreen_ctx", nullptr, nullptr);
  glfwDefaultWindowHints();
  if (!offscreen_ctx) { return false; }
  glfwMakeContextCurrent(offscreen_ctx);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { return false; }
  m_window = offscreen_ctx;
  m_is_initialized = true;
  return true;
}

void hikari::GraphicsOpenGLInstance::terminate()
{
  if (!m_is_initialized) { return; }
  m_is_initialized = false;
  glfwDestroyWindow(m_window);
  m_window = nullptr;
}

bool hikari::GraphicsOpenGLInstance::isInitialized() const
{
  return m_is_initialized;
}
