#include <hikari/core/spectrum.h>

auto hikari::core::SpectrumObject::getRGBColor(ColorSpace to_color_space, Bool is_linear) const -> ColorRGB
{
  auto xyz = getXYZColor();
  if (to_color_space == ColorSpace::eDefault) {
    auto setting = ColorSetting(getColorSetting());
    to_color_space = setting.getDefaultColorSpace();
  }
  return convertXYZ2RGB(xyz, to_color_space);
}
