#pragma once
#include <hikari/render/gfx_instance.h>
namespace hikari {
  inline namespace platforms {
    namespace opengl {
      inline namespace render {
        struct GFXOpenGLWindowObject;
        struct GFXOpenGLInstanceObject : public hikari::render::GFXInstanceObject, public std::enable_shared_from_this<GFXOpenGLInstanceObject> {
          static auto create() -> std::shared_ptr< GFXOpenGLInstanceObject>;
          GFXOpenGLInstanceObject() noexcept;
          virtual ~GFXOpenGLInstanceObject() noexcept;
          auto getAPI() const->GFXAPI override { return GFXAPI::eOpenGL; }
          auto createOpenGLWindow(const GFXWindowDesc& desc) -> std::shared_ptr<GFXOpenGLWindowObject>;
          auto createWindow(const GFXWindowDesc& desc) -> std::shared_ptr<hikari::render::GFXWindowObject> override;
          auto createDevice(const GFXDeviceDesc& desc, const std::shared_ptr<GFXWindowObject>& window = nullptr) -> std::shared_ptr<GFXDeviceObject> override { return nullptr; }
          auto getMainContext() const -> void*;
        private:
          void* m_main_window;
        };
        struct GFXOpenGLWindow;
        struct GFXOpenGLInstance : protected GFXInstanceImpl<GFXOpenGLInstanceObject> {
          using impl_type = GFXInstanceImpl<GFXOpenGLInstanceObject>;
          using type = typename impl_type::type;
          GFXOpenGLInstance() noexcept :impl_type(GFXOpenGLInstanceObject::create()) {}
          GFXOpenGLInstance(const GFXOpenGLInstance& v) noexcept : impl_type(v.getObject()) {}
          GFXOpenGLInstance(nullptr_t) noexcept :impl_type(nullptr) {}
          GFXOpenGLInstance(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
          template<typename GFXOpenGLInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXOpenGLInstanceLike::type>, nullptr_t > = nullptr>
          GFXOpenGLInstance(const GFXOpenGLInstanceLike& v) noexcept :impl_type(v.getObject()) {}
          GFXOpenGLInstance& operator=(const GFXOpenGLInstance& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
          GFXOpenGLInstance& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
          GFXOpenGLInstance& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
          template<typename GFXOpenGLInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXOpenGLInstanceLike::type>, nullptr_t > = nullptr>
          GFXOpenGLInstance& operator=(const GFXOpenGLInstanceLike& v) noexcept { setObject(v.getObject()); return*this; }
          auto createWindow(const GFXWindowDesc& desc) -> GFXOpenGLWindow;

          using impl_type::operator!;
          using impl_type::operator bool;
          using impl_type::getObject;
        };
      }
    }
  }
}
