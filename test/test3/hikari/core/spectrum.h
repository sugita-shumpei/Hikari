#pragma once
#include <hikari/core/color.h>
#include <glm/glm.hpp>
namespace hikari {
  inline namespace core {
    // スペクトル
    struct Spectrum {

    };

    inline auto convertColorXYZToSpectrum(const ColorXYZ& xyz) -> Spectrum { return {}; }
    inline auto convertColorRGBToSpectrum(const ColorRGB& rgb, ColorSpace rgbColorSpace = ColorSpace::eCIE1931) -> Spectrum { return {}; }
  }
}
