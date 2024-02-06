#include <cstdio>
#include <slang.h>
#include <slang-gfx.h>
#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <d3d12.h>
#include <vector>
#include "test2_config.h"

struct MyDebugCallback : public gfx::IDebugCallback
{
  virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(
    gfx::DebugMessageType type,
    gfx::DebugMessageSource source,
    const char* message) override
  {
    printf("%s\n", message);
  }
};

static MyDebugCallback gCallback;

int main() {
  gfx::gfxEnableDebugLayer();
  gfx::gfxSetDebugCallback(&gCallback);
  const auto device_type = gfx::DeviceType::DirectX12;

  Slang::ComPtr<gfx::IDevice> device;
  auto adapters           = gfx::gfxGetAdapters(device_type);
  if (adapters.getCount() == 0) {
    fprintf(stderr, "Failed To Enumerate GFX Adapters!\n");
    return -1;
  }

  gfx::IDevice::Desc device_desc  = {};
  device_desc             = {};
  device_desc.deviceType  = device_type;
  device_desc.adapterLUID = &adapters.getAdapters()[0].luid;

  if (!SLANG_SUCCEEDED(gfx::gfxCreateDevice(&device_desc, device.writeRef()))) {
    fprintf(stderr, "Failed To Create GFX Device!\n");
    return -1;
  }


  return 0;
}
