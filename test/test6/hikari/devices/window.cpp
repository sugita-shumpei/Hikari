#include <hikari/devices/window.h>
#include <hikari/events/system.h>
#include <hikari/events/queue.h>
#include <hikari/events/window_event.h>
using namespace hikari;

auto hikari::Window::getNativeWindow() const -> void*
{
  return m_handle;
}

auto hikari::Window::getSize() const -> UVec2
{
  return m_data.size;
}

auto hikari::Window::getSurfaceSize() const -> UVec2
{
  return m_data.surface_size;
}

auto hikari::Window::getPosition() const -> IVec2
{
  return m_data.position;
}

void hikari::Window::update()
{
  
}

hikari::Window::Window(const WindowCreateDesc& desc) noexcept
  :m_handle{ nullptr },m_data {
    this,
    {desc.width,desc.height},
    { desc.width,desc.height },
    {desc.x,desc.y},
    desc.title,
    desc.flags
  },
  m_target{},
  m_will_destroy{false}
{

  if (m_data.flags & FlagBits::eGraphicsOpenGL) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(HK_DEBUG)
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT , GLFW_TRUE);
#endif
  }
  else {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }

  glfwWindowHint(GLFW_FLOATING , (m_data.flags & FlagBits::eFloating));
  glfwWindowHint(GLFW_RESIZABLE, (m_data.flags & FlagBits::eResizable));
  glfwWindowHint(GLFW_VISIBLE  , (m_data.flags & FlagBits::eVisible));

  if (m_data.flags & FlagBits::eFullscreen) {
    // FULLSCREEN OPERATION
  }
  
  auto window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
  m_handle = window;
  glfwSetWindowUserPointer(m_handle, &m_data);
  glfwSetWindowSizeCallback(m_handle, [](GLFWwindow* handle, int w, int h) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    p_data->size = { w,h };
    if (p_data->window->willDestroy()) { return; }
    // Resize Event
    auto& queue = EventSystem::getInstance()->getGlobalQueue();
    queue.push(std::make_unique<WindowResizeEvent>(p_data->window, UVec2{ w,h }));
  });
  glfwSetFramebufferSizeCallback(m_handle, [](GLFWwindow* handle, int w, int h) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    p_data->surface_size = { w,h };
    if (p_data->window->willDestroy()) { return; }
    // Resize Event
  });
  glfwSetWindowPosCallback(m_handle, [](GLFWwindow* handle, int x, int y) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    p_data->position = { x,y };
    if (p_data->window->willDestroy()) { return; }
    // Moved Event
    auto& queue = EventSystem::getInstance()->getGlobalQueue();
    queue.push(std::make_unique<WindowMovedEvent>(p_data->window, IVec2{ x,y }));
  });
  glfwSetWindowCloseCallback(m_handle, [](GLFWwindow* handle) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
    // Resize Event
    auto& queue  = EventSystem::getInstance()->getGlobalQueue();
    queue.push(std::make_unique<WindowCloseEvent>(p_data->window));
  });
  glfwSetKeyCallback(m_handle, [](GLFWwindow* handle,int key, int scancode,int action, int mods) {
    auto* p_data   = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
    auto key_input = convertInt2KeyInput(key);
    auto key_state = convertInt2KeyState(action);
    auto key_mods  = convertInt2KeyMods(mods);
    // 3つのイベントを発行する
    // 1. Key Down  : キーが、初めてダウン状態
    // 2. Key Up    : キーが、初めてアップ状態
    // 3. Key Repeat: キーが、継続＋ダウン状態
    if (key_state == KeyStateFlagsPress  ){}
    if (key_state == KeyStateFlagsRelease) {}
    if (key_state == KeyStateFlagsRepeat ) {}
  });
  glfwSetCharCallback(m_handle, [](GLFWwindow* handle, unsigned int keycode) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
    auto key     = static_cast<char32_t>(keycode);// UTF32
  });
  glfwSetMouseButtonCallback(m_handle, [](GLFWwindow* handle, int button, int action, int mods) {
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
    auto mouse_input = convertInt2MouseButtonInput(button);
    auto mouse_state = convertInt2KeyState(action);
    auto key_mods    = convertInt2KeyMods(mods);
    // 3つのイベントを発行する
    // 1. Mouse Button Down  : マウスボタンが、ダウン状態
    // 2. Mouse Button Up    : マウスボタンが、アップ状態
    if (mouse_state == KeyStateFlagsPress  ) {}
    if (mouse_state == KeyStateFlagsRelease) {}
  });
  glfwSetScrollCallback(m_handle, [](GLFWwindow* handle, double xoff, double yoff){
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
  });
  glfwSetCursorPosCallback(m_handle, [](GLFWwindow* handle, double x, double y){
    auto* p_data = reinterpret_cast<Window::Data*>(glfwGetWindowUserPointer(handle));
    if (p_data->window->willDestroy()) { return; }
  });

  m_target = EventSystem::getInstance()->createTarget("windows[" + desc.title + "]");
  m_target.addHandler(makeUniqueEventHandler<WindowDestroyEvent>([this](const WindowDestroyEvent& e) {
      auto window = e.getWindow();
      if (this == window) {
        onDestroy();
      }
    }
  ));
}

hikari::Window::~Window() noexcept
{
  EventSystem::getInstance()->destroyTarget(m_target);
}

void hikari::Window::onDestroy()
{
  m_will_destroy = true;// will destroyになるとwindow eventが呼ばれなくなる
}

bool hikari::Window::willDestroy() const noexcept
{
  return m_will_destroy;
}

