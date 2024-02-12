#pragma once
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    enum class ResourceFormat {
      // 8bit
      eR8_SRGB,
      eR8_UNORM,
      eR8_SNORM,
      eR8_UINT,
      eR8_SINT,
      //16bit
      eR8G8_SRGB,
      eR8G8_UNORM,
      eR8G8_SNORM,
      eR8G8_UINT,
      eR8G8_SINT,
      eR16_SRGB,
      eR16_UNORM,
      eR16_SNORM,
      eR16_UINT,
      eR16_SINT,
      eR16_FLOAT,
      //24bit
      eR8G8B8_SRGB,
      eR8G8B8_UNORM,
      eR8G8B8_SNORM,
      eR8G8B8_UINT,
      eR8G8B8_SINT,
      eB8G8R8_SRGB,
      eB8G8R8_UNORM,
      eB8G8R8_SNORM,
      eB8G8R8_UINT,
      eB8G8R8_SINT,
      //32bit
      eR8G8B8A8_SRGB,
      eR8G8B8A8_UNORM,
      eR8G8B8A8_SNORM,
      eR8G8B8A8_UINT,
      eR8G8B8A8_SINT,
      eB8G8R8A8_SRGB,
      eB8G8R8A8_UNORM,
      eB8G8R8A8_SNORM,
      eB8G8R8A8_UINT,
      eB8G8R8A8_SINT,
      eR16G16_SRGB,
      eR16G16_UNORM,
      eR16G16_SNORM,
      eR16G16_UINT,
      eR16G16_SINT,
      eR16G16_FLOAT,
      eR32_UINT,
      eR32_SINT,
      eR32_FLOAT,
      // 48bit
      eR16G16B16_SRGB,
      eR16G16B16_UNORM,
      eR16G16B16_SNORM,
      eR16G16B16_UINT,
      eR16G16B16_SINT,
      eR16G16B16_FLOAT,
      // 64bit
      eR16G16B16A16_SRGB,
      eR16G16B16A16_UNORM,
      eR16G16B16A16_SNORM,
      eR16G16B16A16_FLOAT,
      eR16G16B16A16_FLOAT,
      eR32G32_UINT,
      eR32G32_SINT,
      eR32G32_FLOAT,
      // 96bit
      eR32G32B32_UINT,
      eR32G32B32_SINT,
      eR32G32B32_FLOAT,
      // 128bit
      eR32G32B32A32_UINT,
      eR32G32B32A32_SINT,
      eR32G32B32A32_FLOAT,
      // Compress
      eRGBA_DXT1_SRGB,
      eRGBA_DXT1_UNORM,
      eRGBA_DXT3_SRGB,
      eRGBA_DXT3_UNORM,
      eRGBA_DXT5_SRGB,
      eRGBA_DXT5_UNORM,
      eR_BC4_UNORM,
      eR_BC4_SNORM,
      eRG_BC5_UNORM,
      eRG_BC5_SNORM,
      eRGB_BC6H_UFLOAT,
      eRGB_BC6H_SFLOAT,
      eRGB_BC7_SRGB,
      eRGB_BC7_UNORM,
    };
    enum class ResourceType {
      eBuffer ,
      eImage,
    };
    struct     ResourceObject : public Object {
      virtual ~ResourceObject() noexcept {}
    };
    struct     Resource {

    };
  }
}
