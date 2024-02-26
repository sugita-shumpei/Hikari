#include <hikari/window/application.h>

int hikari::WindowApplication::run(int argc, const char* argv[]) {
  if (initialize()) {
    mainLoop();
  }
  else {
    return -1;
  }
  terminate();
  return 0;
}

bool hikari::WindowApplication::initialize() {
  if (!initSystem()) { return false; }
  if (!initGraphics()) { return false; }
  if (!initWindow()) { return false; }
  if (!initRenderThread()) { return false; }
  if (!initUI()) { return false; }
  return true;
}

void hikari::WindowApplication::terminate() {
  freeUI();
  freeRenderThread();
  freeWindow();
  freeGrapihcs();
  freeSystem();
}

void hikari::WindowApplication::mainLoop() {
  auto render_loop = m_render_thread->submit_task([this]() {this->renderLoop(); });
  eventsLoop();
  render_loop.wait();
}

bool hikari::WindowApplication::initSystem() {
  return WindowSystem::getInstance().initialize();
}

void hikari::WindowApplication::freeSystem() {
  WindowSystem::getInstance().terminate();
}

bool hikari::WindowApplication::initGraphics() {
  return GraphicsSystem::getInstance().initGraphics(m_graphics_api);
}

void hikari::WindowApplication::freeGrapihcs() {
  GraphicsSystem::getInstance().freeGraphics(m_graphics_api);
}

bool hikari::WindowApplication::initWindow() {
  auto& window_system = WindowSystem::getInstance();
  auto main_window = window_system.createWindow("title",800,600,300,300,GraphicsAPIType::eOpenGL,false,true,true,false);
  if (!main_window) { return false; }
  m_main_window = main_window;
  return true;
}

void hikari::WindowApplication::freeWindow() {
  if (!m_main_window) { return; }
  auto& window_system = WindowSystem::getInstance();
  window_system.destroyWindow(m_main_window);
  m_main_window = nullptr;
}

bool hikari::WindowApplication::initRenderThread() {
  m_render_thread = std::make_unique<BS::thread_pool>(1, [this]() {
    auto handle = m_main_window->getHandle();
    if (m_graphics_api == GraphicsAPIType::eOpenGL) { glfwMakeContextCurrent(handle); }
    return;
  });
  return true;
}

void hikari::WindowApplication::freeRenderThread() {
  m_render_thread.reset();
}

bool hikari::WindowApplication::initUI() {
  auto ctx = ImGui::CreateContext();
  if (!ctx) { return false; }
  (void)ImGui::GetIO();
  ImGui::StyleColorsDark();

  if (m_graphics_api == GraphicsAPIType::eOpenGL) {
    auto handle = m_main_window->getHandle();
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    auto init = m_render_thread->submit_task([&]()->bool {
      return ImGui_ImplOpenGL3_Init("#version 460 core");
      });
    if (!init.get()) { return false; }
  }
  if (m_graphics_api == GraphicsAPIType::eVulkan) {
    auto handle = m_main_window->getHandle();
    ImGui_ImplGlfw_InitForVulkan(handle, true);
  }
  return true;
}

void hikari::WindowApplication::freeUI() {
  auto free = m_render_thread->submit_task([&]()->void { ImGui_ImplOpenGL3_Shutdown(); }); free.get();
  ImGui_ImplGlfw_Shutdown();
}

void hikari::WindowApplication::eventsLoop() {
  {
    while (m_is_running) {
      // 同期可能を待機
      m_ready_sync.lock();
      syncFrame();
      // 同期終了を通知
      m_finish_sync.unlock();
      // 入力受付は同期をとる必要なし
      processInput();
      // 入力を処理する
      eventsFrame();
    }
  }
}

void hikari::WindowApplication::renderLoop() {
  while (m_is_running) {
    m_finish_sync.lock();
    // 同期終了を待機
    renderFrame();
    // 同期可能を待機
    m_ready_sync.unlock();
  }
}

void hikari::WindowApplication::syncFrame() {// GLFWのWindow関数はメインスレッドから呼び出す必要あり
  // 受け付けた入力値を更新する
  m_main_window->update();
  m_is_running = !m_main_window->isClose();
  ImGui_ImplGlfw_NewFrame();
}

void hikari::WindowApplication::eventsFrame()
{
  // F11 KEYが設定されていたら画面を最大化する
  auto key_f11 = m_main_window->getKey(KeyInput::eF11);
  if (key_f11 == (KeyStateFlagBits::ePress | KeyStateFlagBits::eUpdate)) {
    m_main_window->setFullscreen(!m_main_window->isFullscreen());
  }
}

void hikari::WindowApplication::processInput() {
  glfwPollEvents();
}

void hikari::WindowApplication::renderFrame() {
  renderUpdate();
  renderMain();
  renderUI();
  // OpenGLの場合, ここでスワップを行う
  if (m_graphics_api == GraphicsAPIType::eOpenGL) {
    auto handle = m_main_window->getHandle();
    glfwSwapBuffers(handle);
  }
}

void hikari::WindowApplication::renderUpdate() {
  // IMGUIの描画に必要なグラフィックスリソースのセットアップ
  if (m_graphics_api == GraphicsAPIType::eOpenGL) {
    ImGui_ImplOpenGL3_NewFrame();
  }
  static float x = 0.0f;
  static float y = 0.0f;
  // IMGUIの処理(書き換えられるように)
  ImGui::NewFrame();
  ImGui::Begin("Hello, world!");
  ImGui::Text("This is some useful text.");
  ImGui::DragFloat("x", &x);
  ImGui::DragFloat("y", &y);
  ImGui::End();
  ImGui::Render();
}

void hikari::WindowApplication::renderMain() {
  // 描画処理
  glClear(GL_COLOR_BUFFER_BIT);
  glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
}

void hikari::WindowApplication::renderUI() {
  // IMGUIの描画コマンドをグラフィックスへ送信
  if (m_graphics_api == GraphicsAPIType::eOpenGL) {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }
}
