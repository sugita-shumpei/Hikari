#pragma once
#include <hikari/render/gfx_window.h>
#include <hikari/platform/glfw/render/gfx_glfw_window.h>
#include <hikari/platform/opengl/render/gfx_opengl_instance.h>
namespace hikari {
  inline namespace platforms {
    namespace opengl {
      inline namespace render {
        struct GFXOpenGLWindowObject : public hikari::platforms::glfw::render::GFXGLFWWindowObject {
          static auto create(const std::shared_ptr<GFXOpenGLInstanceObject>& instance, const GFXWindowDesc& desc) -> std::shared_ptr<GFXOpenGLWindowObject>;
          virtual ~GFXOpenGLWindowObject() noexcept;
          auto getAPI() const->GFXAPI override { return GFXAPI::eOpenGL; }
        private:
          GFXOpenGLWindowObject(const std::shared_ptr<GFXOpenGLInstanceObject>& instance, GLFWwindow* window, const GFXWindowDesc& desc);
        private:
          std::shared_ptr<GFXOpenGLInstanceObject> m_instance;
        };
        struct GFXOpenGLWindow : protected GFXWindowImpl<GFXOpenGLWindowObject> {
          HK_RENDER_GFX_WINDOW_METHOD_OVERRIDES(GFXWindowImpl<GFXOpenGLWindowObject>);
          using type = typename impl_type::type;
          GFXOpenGLWindow() noexcept :impl_type() {}
          GFXOpenGLWindow(const GFXOpenGLWindow& v) noexcept : impl_type(v.getObject()) {}
          GFXOpenGLWindow(nullptr_t) noexcept :impl_type(nullptr) {}
          GFXOpenGLWindow(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
          template<typename GFXOpenGLWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXOpenGLWindowLike::type>, nullptr_t > = nullptr>
          GFXOpenGLWindow(const GFXOpenGLWindowLike& v) noexcept :impl_type(v.getObject()) {}
          GFXOpenGLWindow& operator=(const GFXOpenGLWindow& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
          GFXOpenGLWindow& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
          GFXOpenGLWindow& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
          template<typename GFXOpenGLWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXOpenGLWindowLike::type>, nullptr_t > = nullptr>
          GFXOpenGLWindow& operator=(const GFXOpenGLWindowLike& v) noexcept { setObject(v.getObject()); return*this; }
        };
      }
    }
  }
}
