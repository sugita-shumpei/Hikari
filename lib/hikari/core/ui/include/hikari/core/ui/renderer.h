#pragma once
#include <hikari/core/ui/common.h>
#include <BS_thread_pool.hpp>
namespace hikari {
  namespace core {
    struct Window;
    struct WindowRenderer;
    // UIの描画用クラス
    struct UIManager;
    struct UIRenderer {
      friend struct UIManager;
      UIRenderer(Window* window);
      virtual ~UIRenderer() noexcept;
      virtual bool initialize();// 初期化
      virtual void terminate();// 終了処理
      virtual void update();// 更新
      auto getWindow()   -> Window*;
      auto getRenderer() -> WindowRenderer*;
    protected:
      auto getRenderThread() -> BS::thread_pool*;
      template<typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
      auto executeInRenderThread(F&& callback) {
        return getRenderThread()->submit_task(callback);
      }
      void setManager(UIManager* manager);
      auto getManager()  -> UIManager*;
      auto getDrawData() -> ImDrawData&;
    private:
      Window*    m_window;
      UIManager* m_manager;
      ImDrawData m_draw_data;
    };
  }
}
