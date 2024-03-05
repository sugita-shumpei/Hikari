#pragma once
#include <hikari/core/window/renderer.h>
#include <hikari/core/graphics/opengl/context.h>
namespace hikari {
  namespace core {
    struct GraphicsOpenGLContext;
    struct GraphicsOpenGLRenderer : public WindowRenderer {
      friend struct GraphicsOpenGLContext;
      friend struct GraphicsOpenGLUIRenderer;
      GraphicsOpenGLRenderer(Window* window);
      virtual ~GraphicsOpenGLRenderer() noexcept {}
      bool initialize() override final;
      void terminate() override final;
      void update() override final;
    protected:
      auto getRenderContext() -> GraphicsOpenGLContext*;
      auto createSharingContext() -> GraphicsOpenGLContext*;
      void destroySharingContext(GraphicsOpenGLContext* context);
      inline void swapInterval(int interv)const { glSwapInterval(interv); }
      inline void setContextCurrent(void* context) const { glSetCurrentContext(context); }
      inline void setContextCurrent()const { glSetCurrentContext(); }
    private:
      bool initContext();
      void freeContext();
      virtual bool initCustom();
      virtual void freeCustom();
      virtual void updateCustom();
      inline void  swapBuffers() { glSwapBuffers(); }
    private:
      std::unique_ptr<GraphicsOpenGLContext> m_render_context;
      std::unique_ptr<GraphicsOpenGLContext> m_window_context;
      bool m_is_init         = false;
      bool m_is_init_context = false;
      bool m_is_init_extra   = false;
    };
  }
}
