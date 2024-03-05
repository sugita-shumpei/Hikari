#pragma once
#include <hikari/core/ui/renderer.h>
namespace hikari {
  namespace core {
    struct GraphicsOpenGLUIRenderer : public UIRenderer{
      GraphicsOpenGLUIRenderer(Window* window);
      virtual ~GraphicsOpenGLUIRenderer() noexcept {}
      bool initialize()override;// 初期化
      void terminate()override;// 終了処理
      void update()override;

    };
  }
}
