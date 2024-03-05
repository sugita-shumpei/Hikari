#pragma once
#include <BS_thread_pool.hpp>
#include <hikari/core/event/manager.h>
#include <hikari/core/input/input.h>
#include <hikari/core/window/common.h>
#include <hikari/core/window/app.h>
namespace hikari {
  namespace core {
    struct WindowApp;
    struct WindowRenderer;
    struct Window {
      Window(WindowApp* app, const WindowDesc& desc);
      virtual ~Window() noexcept;
      // App
      auto getApp() -> WindowApp*;
      auto getRenderer() -> WindowRenderer* { return m_renderer.get(); };
      // Getter
      auto getSize() const -> WindowExtent2D;
      auto getSurfaceSize() const -> WindowExtent2D;
      auto getPosition() const -> WindowOffset2D;
      auto getNativeHandle() -> void*;
      auto getTitle() const -> const std::string&;// タイトルの取得
      // Setter
      void setSize(const WindowExtent2D& size);
      void setPosition(const WindowOffset2D& pos);
      void setTitle(const std::string& title);
      void setFullscreen(bool is_fullscreen);
      void setRenderer(std::unique_ptr<WindowRenderer>&& renderer);;
      // Boolean
      bool isClosed() const;// 閉じたかどうか
      bool isFocused() const;// フォーカスしているかどうか
      bool isHovered() const;// Cursorが画面上にあるかどうか
      bool isIconified() const;// 最小化されているかどうか
      bool isFullscreen() const;// フルスクリーンかどうか
      bool isVisible() const;// 可視かどうか
      bool isResizable() const;// リサイズかどうか
      bool isFloating() const;// Floatingかどうか
      bool isGraphicsOpenGL() const;
      bool isGraphicsVulkan() const;
      // Inputs
      auto getInput() const noexcept -> const Input&;
    private:
      friend class WindowApp;
      void updateInput();
      static void pollEvents();
      void onSync();
    private:
      friend class WindowRenderer;
      // WindowManager(Resize,Position...)
      auto getWindowEventManager() noexcept -> EventManager&;
      // InputManager(KeyPress...)
      auto getInputEventManager() noexcept -> EventManager&;
      auto getRenderThread() noexcept -> BS::thread_pool*
      {
        return m_app->getRenderThread();
      }
      template<typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
      auto executeInRenderThread(F&& callback) {
        return getRenderThread()->submit_task(callback);
      }
    private:
      WindowApp* m_app;
      void*      m_handle;
      WindowData m_data;
      std::unique_ptr<WindowRenderer> m_renderer = nullptr;

    };
  }
}
