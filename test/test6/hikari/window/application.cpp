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
  if (m_graphics_api == GraphicsAPIType::eOpenGL) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  }
  else {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  GLFWwindow* main_window = glfwCreateWindow(800, 600, "main_window", nullptr, nullptr);
  glfwDefaultWindowHints();
  if (!main_window) { return false; }
  m_main_window = main_window;
  return true;
}

void hikari::WindowApplication::freeWindow() {
  glfwDestroyWindow(m_main_window);
  m_main_window = nullptr;
}

bool hikari::WindowApplication::initRenderThread() {
  m_render_thread = std::make_unique<BS::thread_pool>(1, [this]() {
    if (m_graphics_api == GraphicsAPIType::eOpenGL) { glfwMakeContextCurrent(m_main_window); }
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
    ImGui_ImplGlfw_InitForOpenGL(m_main_window, true);
    auto init = m_render_thread->submit_task([&]()->bool {
      return ImGui_ImplOpenGL3_Init("#version 460 core");
      });
    if (!init.get()) { return false; }
  }
  if (m_graphics_api == GraphicsAPIType::eVulkan) {
    ImGui_ImplGlfw_InitForVulkan(m_main_window, true);
  }
  return true;
}

void hikari::WindowApplication::freeUI() {
  auto free = m_render_thread->submit_task([&]()->void {
    ImGui_ImplOpenGL3_Shutdown();
  });
  free.get();
  ImGui_ImplGlfw_Shutdown();
}

void hikari::WindowApplication::eventsLoop() {
  {
    while (m_is_running) {
      // 同期可能を待機
      {
        std::unique_lock<std::mutex> lk(m_mtx_ready_sync);
        m_cv_ready_sync.wait(lk, [this]() { return m_is_ready_sync; });
        m_is_ready_sync = false;
      }
      syncFrame();
      // 同期終了を通知
      {
        std::lock_guard<std::mutex> lk(m_mtx_finish_sync);
        m_is_finish_sync = true;
        m_cv_finish_sync.notify_one();
      }
      // 入力イベントは同期をとる必要なし
      processInput();
      // 非同期IO: 別スレッドで大きなテキストファイルの読み込みを行う
      eventsFrame();
    }
  }
}

void hikari::WindowApplication::renderLoop() {
  while (m_is_running) {
    // 同期終了を待機
    {
      std::unique_lock<std::mutex> lk(m_mtx_finish_sync);
      m_cv_finish_sync.wait(lk, [this]() { return m_is_finish_sync; });
      m_is_finish_sync = false;
    }
    renderFrame();
    // 同期可能を通知
    {
      std::lock_guard<std::mutex> lk(m_mtx_ready_sync);
      m_is_ready_sync = true;
      m_cv_ready_sync.notify_one();
    }
  }
}

void hikari::WindowApplication::syncFrame() {// GLFWのWindow関数はメインスレッドから呼び出す必要あり
  m_is_running = !glfwWindowShouldClose(m_main_window);
  ImGui_ImplGlfw_NewFrame();
}

void hikari::WindowApplication::eventsFrame()
{
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
    glfwSwapBuffers(m_main_window);
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
