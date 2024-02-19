#pragma once
#include <hikari/render/gfx_common.h>
namespace hikari {
  inline namespace render {
    struct GFXDeviceObject {
      virtual ~GFXDeviceObject() noexcept {}
      virtual auto getAPI() const->GFXAPI = 0;
    };
    template<typename GFXDeviceObjectT, std::enable_if_t<std::is_base_of_v<GFXDeviceObject, GFXDeviceObjectT>, nullptr_t> = nullptr>
    struct GFXDeviceImpl : protected GFXWrapper<GFXDeviceObjectT> {
      using impl_type = GFXWrapper<GFXDeviceObjectT>;
      using type = typename impl_type::type;
      GFXDeviceImpl() noexcept :impl_type() {}
      GFXDeviceImpl(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXDeviceImpl(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXDeviceLike, std::enable_if_t<std::is_base_of_v<GFXDeviceObjectT, typename GFXDeviceLike::type>, nullptr_t > = nullptr>
      GFXDeviceImpl(const GFXDeviceLike& v) noexcept :impl_type(v.getObject()) {}
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::getObject;
      using impl_type::getAPI;
    protected:
      using impl_type::setObject;
    };
    struct GFXWindow;
    struct GFXDevice : protected GFXDeviceImpl<GFXDeviceObject> {
      using impl_type = GFXDeviceImpl<GFXDeviceObject>;
      using type = typename impl_type::type;
      GFXDevice() noexcept :impl_type() {}
      GFXDevice(const GFXDevice& v) noexcept : impl_type(v.getObject()) {}
      GFXDevice(nullptr_t) noexcept :impl_type(nullptr) {}
      GFXDevice(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
      template<typename GFXDeviceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXDeviceLike::type>, nullptr_t > = nullptr>
      GFXDevice(const GFXDeviceLike& v) noexcept :impl_type(v.getObject()) {}
      GFXDevice& operator=(const GFXDevice& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
      GFXDevice& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
      GFXDevice& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
      template<typename GFXDeviceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXDeviceLike::type>, nullptr_t > = nullptr>
      GFXDevice& operator=(const GFXDeviceLike& v) noexcept { setObject(v.getObject()); return*this; }
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::getObject;
    };
  }
}
