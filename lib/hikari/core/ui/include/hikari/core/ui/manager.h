#pragma once
#include <memory>
#include <hikari/core/ui/common.h>
#include <hikari/core/ui/renderer.h>
namespace hikari {
  namespace core {
    struct Window;
    struct UIRenderer;
    struct UIManager {
      UIManager(UIRenderer* renderer);
      virtual ~UIManager() noexcept{}
      auto getRenderer() -> UIRenderer* { return m_renderer.get(); }
      bool initialize();
      void terminate();
      void update();
      void render();
      void onSync();
      auto getWindow() -> Window*;
      bool isHoveredMouse() const noexcept;
    private:
      bool initUI();
      void freeUI();
      bool initRenderer();
      void freeRenderer();
      virtual void onUpdate();
    private:
      std::unique_ptr<UIRenderer> m_renderer;
      bool m_is_init          = false;
      bool m_is_init_ui       = false;
      bool m_is_init_renderer = false;
    };
  }
}
