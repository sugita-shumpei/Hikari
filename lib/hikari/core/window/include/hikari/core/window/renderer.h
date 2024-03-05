#pragma once
#include <hikari/core/window/common.h>
#include <hikari/core/window/window.h>

namespace hikari {
  namespace core {
    struct Window;
    struct WindowApp;
    using  Window_PFN_glProcAddr = void(*)(void);
    struct WindowRenderer {
      friend class Window;
      friend class UIManager;
      friend class UIRenderer;
       WindowRenderer(Window* window);
      ~WindowRenderer() noexcept;
      virtual bool initialize() { return true; };// 初期化
      virtual void terminate() {};// 終了処理
      virtual void update();// 更新する
      auto getSurfaceSize() const noexcept -> WindowExtent2D;// どのタイミングで更新するか?
      auto getWindow() const noexcept -> Window* { return m_window; }
    protected:
      void glSwapInterval(int interval)const;// OpenGL専用: 垂直同期のインターバルを指定(かならずcurrentなコンテキストから呼ぶこと)
      auto glCreateSharingContext() const -> void*;
      void glDestroySharingContext(void*)const;
      void glSetCurrentContext(void*)const;
      void glSetCurrentContext()const;
      auto glGetProcAddress() -> Window_PFN_glProcAddr(*)(const char*);
      void glSwapBuffers();// OpenGL専用: 画面をスワップする
      auto vkCreateSurface(void* instance) -> void*;// Vulkan専用: サーフェスを作成する
      auto getRenderThread() noexcept -> BS::thread_pool*;
      // OpenGLなどは特定のWindow, 特定のスレッドにコンテキストが紐づいているため,
      // 初期化処理などはRender Threadで呼び出す必要あり
      template<typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
      auto executeInRenderThread(F&& callback) {
        return getRenderThread()->submit_task(callback);
      }
    private:
      virtual void onResize()     {}
      virtual void onFullscreen() {}
      virtual void onWindowed()   {}
      virtual void onSync();//同期処理
    private:
      Window* m_window = {};// window
      WindowExtent2D m_surface_size = {};// 共有変数(sync時に更新)
      bool m_is_fullscreen = false;// 共有変数(sync時に更新)
      bool m_on_resize     = false;// 共有変数(sync時に更新)
      bool m_on_fullscreen = false;// 共有変数(sync時に更新)
      bool m_on_windowed   = false;// 共有変数(sync時に更新)
    };
  }
}
