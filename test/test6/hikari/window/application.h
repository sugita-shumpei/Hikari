#pragma once
#include <thread>
#include <iostream>
#include <atomic>
#include <BS_thread_pool.hpp>
//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
#include <hikari/window/system.h>
#include <hikari/window/window.h>
#include <hikari/graphics/system.h>
#include <hikari/graphics/opengl/common.h>
#include <hikari/thread/spin_lock.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
namespace hikari {
  struct WindowApplication {
    WindowApplication(GraphicsAPIType api = GraphicsAPIType::eOpenGL)noexcept :m_graphics_api{ api } {}
    ~WindowApplication() noexcept {}
    int run(int argc, const char* argv[]);
    bool initialize();
    void terminate();
    void mainLoop();
  private:
    bool initSystem();// WindowSystemの初期化
    void freeSystem();// WindowSystemの開放
    bool initGraphics();// GraphicsSystemの初期化(TODO: 複数Window対応)
    void freeGrapihcs();// GraphicsSystemの開放  (TODO: 複数Window対応)
    bool initWindow();// Windowの作成
    void freeWindow();// Windowの開放
    bool initRenderThread();// レンダースレッドの作成
    void freeRenderThread();// レンダースレッドの開放
    bool initUI();// UISystemの初期化
    void freeUI();// UISystemの開放  
  private:
    void eventsLoop();// (TODO: 複数Window対応)
    void renderLoop();
  private:
    void syncFrame();
    void eventsFrame();
    void processInput();
    void renderFrame();
    void renderUpdate();
    void renderMain();
    void renderUI();
  private:
    GraphicsAPIType m_graphics_api    = GraphicsAPIType::eOpenGL;
    Window* m_main_window = nullptr;
    SpinLock m_ready_sync = false  ; // アンロックの状態で待機する
    SpinLock m_finish_sync = true   ; // 　　ロックの状態で待機する
    bool m_is_running = true;
    std::unique_ptr<BS::thread_pool> m_render_thread   = nullptr;
  };

}
