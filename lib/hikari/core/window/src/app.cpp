#include <hikari/core/window/app.h>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <hikari/core/common/timer.h>
#include <hikari/core/window/window.h>
#include <hikari/core/window/event.h>
#include <hikari/core/window/renderer.h>

hikari::core::WindowApp::WindowApp(int argc, const char* argv[]) noexcept
  :
  hikari::core::App(argc,argv),
  m_window{nullptr},
  m_render_thread{std::make_unique<BS::thread_pool>(1)},
  m_is_init{false},
  m_is_init_window{ false },
  m_is_init_extra{ false },
  m_is_quit{ false},
  m_barrier{2},
  m_window_event_manager()
{
}

hikari::core::WindowApp::~WindowApp() noexcept
{}

void hikari::core::WindowApp::setWindow(Window* window)
{
  if (m_window) { return; }
  m_window.reset(window);
}

auto hikari::core::WindowApp::getWindow() -> Window*
{
  return m_window.get();
}

bool hikari::core::WindowApp::initWindow() {
  try {
    auto desc = hikari::core::WindowDesc();
    desc.position = { 100,100 };
    desc.size = { 800u,600u };
    desc.title = "";
    desc.flags = hikari::core::WindowFlagBits::eVisible| hikari::core::WindowFlagBits::eResizable;
    auto window  = new hikari::core::Window(this,desc);
    setWindow(window);
    return true;
  }
  catch (std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return false;
  }
}

void hikari::core::WindowApp::freeWindow()
{
  m_window.reset();
}

bool hikari::core::WindowApp::initCustom() {
  return true;
}

void hikari::core::WindowApp::freeCustom() {
  
}

bool hikari::core::WindowApp::initialize()
{
  if (m_is_init) { return true; }
  if (!m_is_init_window) {
    if (!initWindow()) { return false; }
    m_is_init_window = true;
  }
  if (!m_is_init_extra) {
    if (!initCustom()) { return false; }
    m_is_init_extra = true;
  }
  m_is_init = true;
  return true;
}

void hikari::core::WindowApp::terminate()
{
  if (m_is_init_extra)  { freeCustom(); }
  if (m_is_init_window) { freeWindow(); }
  m_is_init = false;
}

void hikari::core::WindowApp::mainLoop()
{
  m_window_event_manager.subscribe(makeUniqueEventHandlerFromCallback<WindowResizeEvent>(
    [](const auto& e) { std::cout << "resize! " << std::endl; }
  ));
  m_window_event_manager.subscribe(makeUniqueEventHandlerFromCallback<WindowMovedEvent>(
    [](const auto& e) { std::cout << "moved! " << std::endl; }
  ));
  m_window_event_manager.subscribe(makeUniqueEventHandlerFromCallback<WindowWindowedEvent>(
    [](const auto& e) { std::cout << "windowed! " << std::endl; }
  ));
  m_window_event_manager.subscribe(makeUniqueEventHandlerFromCallback<WindowFullscreenEvent>(
    [](const auto& e) { std::cout << "fullscreen! " << std::endl; }
  ));

  auto future = m_render_thread->submit_task([this]() {
    while (!shouldQuit()) {
      m_barrier.arrive_and_wait();
      this->render();// 描画
      m_barrier.arrive_and_wait();
      m_barrier.arrive_and_wait();
    }
  });

  while (!shouldQuit()) {
    m_barrier.arrive_and_wait();
    this->processInput();// 入力受付
    this->update();// 更新
    m_barrier.arrive_and_wait();
    this->syncFrame();// 同期
    m_barrier.arrive_and_wait();
  }

  future.wait();
}

auto hikari::core::WindowApp::getWindowEventManager() noexcept -> EventManager&
{
  return m_window_event_manager;
}

auto hikari::core::WindowApp::getInputEventManager() noexcept -> EventManager&
{
  return m_input_event_manager;
}

auto hikari::core::WindowApp::getInput() const noexcept -> const Input*
{
  if (m_window->isHovered()) { return &m_window->getInput(); }
  return nullptr;
}

void hikari::core::WindowApp::update()
{
  /////////////////////////////Update関連処理//////////////////////////////////
  auto input = getInput();
  if (input) {
    // 入力を受け取る
    if (input->getKeyPressed(KeyInput::eF11)) {
      m_window->setFullscreen(!m_window->isFullscreen());
    }
    m_window->setTitle("focus");
  }
  else {
    m_window->setTitle("leave");
  }
}

void hikari::core::WindowApp::processInput()
{
  /////////////////////////////PollingEvents//////////////////////////////////
  m_window->updateInput();
  core::Window::pollEvents();
  /////////////////////////////Window関連処理//////////////////////////////////
  m_window_event_manager.dispatchEvents();//<-Window Eventを処理する
  /////////////////////////////Inputs関連処理//////////////////////////////////
  m_input_event_manager.dispatchEvents();//<-Input Eventを処理する
}

void hikari::core::WindowApp::syncFrame()
{
  m_window->onSync();
  this->onSync();
}

void hikari::core::WindowApp::postQuit()
{
  m_is_quit = true;
}

bool hikari::core::WindowApp::shouldQuit()const noexcept
{
  return m_is_quit;
}

void hikari::core::WindowApp::render()
{
  auto renderer = m_window->getRenderer();
  renderer->update();
}

void hikari::core::WindowApp::onSync()
{
  if (m_window->isClosed()) {
    postQuit();
  }
}
