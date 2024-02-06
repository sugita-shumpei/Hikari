#pragma once
#include <glm/glm.hpp>
namespace hikari {
  inline namespace core {
    struct     ColorXYZ  {
      float x;
      float y;
      float z;
    };
    struct     ColorXYZA {
      float r;
      float g;
      float b;
    };
    struct     ColorRGB  {
      float r;
      float g;
      float b;
    };
    struct     ColorRGBA {
      float r;
      float g;
      float b;
      float a;
    };
    enum class ColorSpace {
      eCIE1931 ,
      eSRGB    ,
      eAdobeRGB,
      eBT2020  ,
      eBT2100
    };
    // 色空間の変換を行う
    auto convertColorRGB2XYZ(ColorRGB     rgb, ColorSpace rgbColorSpace = ColorSpace::eCIE1931) -> ColorXYZ;
    auto convertColorXYZ2RGB(ColorXYZ     xyz, ColorSpace rgbColorSpace = ColorSpace::eCIE1931) -> ColorRGB;
    auto convertColorRGBA2XYZA(ColorRGBA rgba, ColorSpace rgbColorSpace = ColorSpace::eCIE1931) -> ColorXYZA;
    auto convertColorXYZA2RGBA(ColorXYZA xyza, ColorSpace rgbColorSpace = ColorSpace::eCIE1931) -> ColorRGBA;
  }
}
