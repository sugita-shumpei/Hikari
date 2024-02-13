#include <hikari/spectrum/blackbody.h>

auto hikari::spectrum::SpectrumBlackbodyObject::getPropertyNames() const -> std::vector<hikari::Str>
{
  return { "min_wavelength","max_wavelength","temperature","color_setting" };
}

void hikari::spectrum::SpectrumBlackbodyObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const
{
  pb.clear();
  pb.setValue("max_wavelength", getMaxWaveLength());
  pb.setValue("min_wavelength", getMinWaveLength());
  pb.setValue("temperature", getTemperature());
  pb.setValue("color_setting", ColorSetting(getColorSetting()));
}

void hikari::spectrum::SpectrumBlackbodyObject::setPropertyBlock(const PropertyBlockBase<Object>& pb)
{
  auto min_wavelength = pb.getValue("min_wavelength").getValueTo<F32>();
  auto max_wavelength = pb.getValue("max_wavelength").getValueTo<F32>();
  auto temperature = pb.getValue("temperature").getValueTo<F32>();
  auto color_setting = pb.getValue("color_setting").getValue<ColorSetting>();
  if (min_wavelength) { setMinWaveLength(*min_wavelength); }
  if (max_wavelength) { setMaxWaveLength(*max_wavelength); }
  if (temperature) { setTemperature(*temperature); }
  if (color_setting) { setColorSetting(color_setting.getObject()); }
}

bool hikari::spectrum::SpectrumBlackbodyObject::hasProperty(const Str& name) const
{
  if (name == "min_wavelength") { return true; }
  if (name == "max_wavelength") { return true; }
  if (name == "temperature") { return true; }
  return false;
}

bool hikari::spectrum::SpectrumBlackbodyObject::getProperty(const Str& name, PropertyBase<Object>& prop) const
{
  if (name == "min_wavelength") { prop.setValue(getMinWaveLength()); return true; }
  if (name == "max_wavelength") { prop.setValue(getMaxWaveLength()); return true; }
  if (name == "temperature") { prop.setValue(getTemperature()); return true; }
  return false;
}

bool hikari::spectrum::SpectrumBlackbodyObject::setProperty(const Str& name, const PropertyBase<Object>& prop)
{
  if (name == "min_wavelength") { auto val = prop.getValueTo<F32>(); if (val) { setMinWaveLength(*val); return true; } return false; }
  if (name == "max_wavelength") { auto val = prop.getValueTo<F32>(); if (val) { setMaxWaveLength(*val); return true; } return false; }
  if (name == "temperature") { auto val = prop.getValueTo<F32>(); if (val) { setTemperature(*val); return true; } return false; }
  if (name == "color_setting") { auto color_setting = prop.getValue<ColorSetting>(); setColorSetting(color_setting.getObject()); return true; }
  return false;
}

auto hikari::spectrum::SpectrumBlackbodyObject::sample(F32 wavelength) const -> F32
{
  if (wavelength < getMinWaveLength()) { return 0.0f; }
  if (wavelength > getMaxWaveLength()) { return 0.0f; }
  return 0.0f;
}

auto hikari::spectrum::SpectrumBlackbodyObject::getRGBColor(ColorSpace to_color_space, Bool is_linear) const -> ColorRGB
{
  auto xyz = getXYZColor();
  if (to_color_space == ColorSpace::eDefault) {
    auto setting = ColorSetting(getColorSetting());
    to_color_space = setting.getDefaultColorSpace();
  }
  auto rgb = convertXYZ2RGB(xyz, to_color_space);
  if (!is_linear) {
    rgb = convertLinearRGB2NonLinearRGB(rgb, to_color_space);
  }
  return rgb;
}

auto hikari::spectrum::SpectrumBlackbodyObject::getXYZColor() const -> ColorXYZ
{
  return ColorXYZ();
}
