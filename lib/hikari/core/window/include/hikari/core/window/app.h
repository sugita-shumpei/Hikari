#pragma once
#include <memory>
#include <BS_thread_pool.hpp>
#include <barrier>
#include <hikari/core/app/app.h>
#include <hikari/core/common/spin_lock.h>
#include <hikari/core/common/timer.h>
#include <hikari/core/event/manager.h>
#include <hikari/core/input/input.h>
namespace hikari {
  namespace core {
    struct Window;
    struct WindowApp :  hikari::core::App {
      friend class Window;
    public:
      WindowApp(int argc, const char* argv[]) noexcept;
      virtual ~WindowApp() noexcept;
    protected:
      void setWindow(Window* window);
      auto getWindow() -> Window*;
      auto getInput() const noexcept -> const Input*;
      void render();
      void postQuit();
      bool shouldQuit()const noexcept;
      virtual void onSync();
      virtual void update();
    private:
      auto getWindowEventManager() noexcept -> EventManager&;
      auto getInputEventManager() noexcept -> EventManager&;
      auto getRenderThread() noexcept -> BS::thread_pool* { return m_render_thread.get(); }
      template<typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
      auto executeInRenderThread(F&& callback) {
        return m_render_thread->submit_task(callback);
      }
    private:
      virtual bool initWindow();
      void freeWindow();
      virtual bool initCustom();
      virtual void freeCustom();
      void processInput();
      void syncFrame();
    private:
      bool initialize() override;
      void terminate() override;
      void mainLoop() override;
    private:
      std::unique_ptr<Window> m_window;
      std::unique_ptr<BS::thread_pool> m_render_thread;
      EventManager m_window_event_manager;
      EventManager m_input_event_manager;
      std::barrier<> m_barrier;
      bool m_is_init;
      bool m_is_init_window;
      bool m_is_init_extra;
      bool m_is_quit;
    };
  }
}
