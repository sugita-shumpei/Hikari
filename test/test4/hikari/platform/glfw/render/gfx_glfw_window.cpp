#include <hikari/platform/glfw/render/gfx_glfw_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
hikari::platforms::glfw::render::GFXGLFWWindowObject::GFXGLFWWindowObject(GLFWwindow* window,const GFXWindowDesc& desc) noexcept
  :
  m_window{window},
  m_focus{false},
  m_close{ false },
  m_visible{ desc.is_visible },
  m_borderless{ desc.is_borderless },
  m_iconified{ false },
  m_resizable{ desc.is_resizable },
  m_size{ desc.width,desc.height },
  m_fb_size{desc.width,desc.height},
  m_position{ 0,0 }
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  int fb_w, fb_h;
  manager.getFramebufferSize(m_window,fb_w, fb_h);
  m_fb_size[0] = fb_w;
  m_fb_size[1] = fb_h;
  int x, y;
  manager.getWindowPosition(m_window, x, y);
  m_position[0] = x;
  m_position[1] = y;
}

hikari::platforms::glfw::render::GFXGLFWWindowObject::~GFXGLFWWindowObject() noexcept
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.destroyWindow(m_window);
}

auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getHandle() const -> void* 
{
  return m_window;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isResizable() const
{
  return m_resizable;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isClosed() const
{
  return m_close;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isFocused() const
{
  return m_focus;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isVisible() const
{
  return m_visible;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isIconified() const
{
  return m_iconified;
}

bool hikari::platforms::glfw::render::GFXGLFWWindowObject::isBorderless() const
{
  return m_borderless;
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setResizable(bool resizable)
{

  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowResizable(getHandle(), resizable);
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setVisible(bool visible)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowVisible(getHandle(), visible);
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setIconified(bool iconified)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowIconified(getHandle());
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setBorderless(bool borderless)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowBorderless(getHandle(),borderless);
  m_borderless = borderless;
}

auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getClipboard() const -> std::string 
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto s = manager.getClipboard(getHandle());
  if (!s) { return ""; }
  else { return std::string(s); }
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setClipboard(const std::string& s)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setClipboard(getHandle(), s.c_str());
}
auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getTitle() const -> std::string
{
  return m_title;
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setTitle(const std::string& s)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowTitle(getHandle(), s.c_str());
  m_title = s;
}

auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getSize() const->std::array<uint32_t, 2> 
{
  return m_size;
}

auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getPosition() const->std::array<uint32_t, 2> 
{
  return m_position;
}

auto hikari::platforms::glfw::render::GFXGLFWWindowObject::getFramebufferSize() const->std::array<uint32_t, 2> 
{
  return m_fb_size;
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setSize(const std::array<uint32_t, 2>& size)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowSize((GLFWwindow*)getHandle(), size[0], size[1]);
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::setPosition(const std::array<uint32_t, 2>& pos)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  manager.setWindowPosition((GLFWwindow*)getHandle(), pos[0], pos[1]);
}

void hikari::platforms::glfw::render::GFXGLFWWindowObject::initEvents()
{
  auto handle = getHandle();
  if (handle) {
    auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
    manager.setWindowUserPointer(handle, this);
#define HK_PLATFORM_GLFW_REGISTER_CALLBACK(NAME) \
    manager.set##Callback##NAME((GLFWwindow*)handle,DefaultCallback##NAME)

    HK_PLATFORM_GLFW_REGISTER_CALLBACK(WindowSize);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(WindowPosition);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(FramebufferSize);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(WindowClose);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(WindowIconified);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(CursorPosition);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(Key);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(Char);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(MouseButton);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(CursorEnter);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(Drop);
    HK_PLATFORM_GLFW_REGISTER_CALLBACK(Scroll);
  }
}


#define HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(NAME,...)   void hikari::platforms::glfw::render::GFXGLFWWindowObject::Default##NAME(GLFWwindow* handle,__VA_ARGS__)
#define HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK_VOID(NAME)  void hikari::platforms::glfw::render::GFXGLFWWindowObject::Default##NAME(GLFWwindow* handle)

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowSize, int32_t w, int32_t h)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  // resize時の処理を行う
  window->m_size[0] = w;
  window->m_size[1] = h;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowPosition, int32_t x, int32_t y)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_position[0] = x;
  window->m_position[1] = y;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK_VOID(CallbackWindowClose)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window   = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_close = true;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowIconified,int32_t iconified)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_iconified = iconified;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackFramebufferSize, int32_t w, int32_t h)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_fb_size[0] = w;
  window->m_fb_size[1] = h;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackCursorPosition, double x, double y)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_cursor_position[0] = x;
  window->m_cursor_position[1] = y;
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackKey, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{

}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackChar, uint32_t codepoint)
{

}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackMouseButton, int32_t button, int32_t action, int32_t mods)
{

}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackCursorEnter, int32_t enter)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  if (enter) {
    window->m_focus = true;
  }
  else {
    window->m_focus = false;
  }
}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackDrop, int path_count, const char* paths[])
{

}

HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackScroll, double x, double y)
{
  auto& manager = hikari::platforms::glfw::WindowManager::getInstance();
  auto window = reinterpret_cast<GFXGLFWWindowObject*>(manager.getWindowUserPointer(handle));
  window->m_scroll[0] = x;
  window->m_scroll[1] = y;
}

