#pragma once
#include <hikari/platform/opengl/render/gfx_opengl_window.h>
namespace hikari {
  inline namespace platforms {
    namespace opengl {
      inline namespace render {
        // OpenGLの処理は残念ながらスレッドに依存している
        // そのため,Vulkanと異なり,RenderThread内で処理を実行する必要あり
        struct GFXOpenGLDevice {

        private:
          // window
          std::shared_ptr<GFXOpenGLWindow> m_window = nullptr;
          // context(offscreen)
          void* m_offscreen_context = nullptr;
        };
      }
    }
  }
}
