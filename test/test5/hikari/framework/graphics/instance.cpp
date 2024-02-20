#include <hikari/render/gfx_instance.h>
#include <hikari/render/gfx_window.h>
#include <hikari/render/gfx_device.h>
auto hikari::render::GFXInstance::createWindow(const GFXWindowDesc& desc) -> GFXWindow {
  auto obj = getObject();
  if (!obj) { return GFXWindow(); }
  return GFXWindow(obj->createWindow(desc));
}

auto hikari::render::GFXInstance::createDevice(const GFXDeviceDesc& desc) -> GFXDevice
{
  auto obj = getObject();
  if (!obj) { return GFXDevice(); }
  return GFXDevice(obj->createDevice(desc));
}

auto hikari::render::GFXInstance::createDevice(const GFXDeviceDesc& desc, const GFXWindow& window) -> GFXDevice
{
  auto obj = getObject();
  if (!obj) { return GFXDevice(); }
  return GFXDevice(obj->createDevice(desc,window.getObject()));
}
