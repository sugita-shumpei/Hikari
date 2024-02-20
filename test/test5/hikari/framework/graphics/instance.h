#pragma once
#include <hikari/render/gfx_common.h>
namespace hikari {
  inline namespace render {
    struct GFXDeviceObject;
    struct GFXWindowObject;
    struct GFXInstanceObject {
      virtual ~GFXInstanceObject() noexcept {}
      virtual auto createWindow(const GFXWindowDesc& desc) -> std::shared_ptr<GFXWindowObject> = 0;
      virtual auto createDevice(const GFXDeviceDesc& desc, const std::shared_ptr<GFXWindowObject>& window = nullptr) -> std::shared_ptr<GFXDeviceObject> = 0;
      virtual auto getAPI() const->GFXAPI = 0;
    };
    template<typename GFXInstanceObjectT, std::enable_if_t<std::is_base_of_v<GFXInstanceObject, GFXInstanceObjectT>, nullptr_t> = nullptr>
    struct GFXInstanceImpl : protected GFXWrapper<GFXInstanceObjectT> {
      using impl_type = GFXWrapper<GFXInstanceObjectT>;
      using type = typename impl_type::type;
      GFXInstanceImpl() noexcept :impl_type() {}
      GFXInstanceImpl(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXInstanceImpl(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXInstanceLike, std::enable_if_t<std::is_base_of_v<GFXInstanceObjectT, typename GFXInstanceLike::type>, nullptr_t > = nullptr>
      GFXInstanceImpl(const GFXInstanceLike& v) noexcept :impl_type(v.getObject()) {}
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::getObject;
      using impl_type::getAPI;
    protected:
      using impl_type::setObject;
    };
    struct GFXDevice;
    struct GFXWindow;
    struct GFXInstance : protected GFXInstanceImpl<GFXInstanceObject> {
      using impl_type = GFXInstanceImpl<GFXInstanceObject>;
      using type = typename impl_type::type;
      GFXInstance() noexcept :impl_type() {}
      GFXInstance(const GFXInstance& v) noexcept : impl_type(v.getObject()) {}
      GFXInstance(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXInstance(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXInstanceLike::type>, nullptr_t > = nullptr>
      GFXInstance(const GFXInstanceLike& v) noexcept :impl_type(v.getObject()) {}
      GFXInstance& operator=(const GFXInstance& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
      GFXInstance& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
      GFXInstance& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
      template<typename GFXInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXInstanceLike::type>, nullptr_t > = nullptr>
      GFXInstance& operator=(const GFXInstanceLike& v) noexcept { setObject(v.getObject()); return*this; }
      auto createWindow(const GFXWindowDesc& desc) -> GFXWindow;
      auto createDevice(const GFXDeviceDesc& desc) -> GFXDevice;
      auto createDevice(const GFXDeviceDesc& desc, const GFXWindow& window) -> GFXDevice;
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::getObject;
    };
  }
}
