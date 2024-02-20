#include <hikari/spectrum/uniform.h>
auto hikari::spectrum::SpectrumUniformObject::sample(F32 wavelength) const -> F32 {
  if (m_min_wavelength <= wavelength && wavelength <= m_max_wavelength) { return m_intensity; }
  return 0.0f;
}

auto hikari::spectrum::SpectrumUniformObject::getXYZColor() const -> ColorXYZ
{
  auto min_wavelength = std::max(m_min_wavelength, 360.0f);
  auto max_wavelength = std::min(m_max_wavelength, 830.0f);
  auto xyz = Vec3();
  auto den = 0.0f;
  for (size_t i = 360; i <= 830 - 1; ++i) {
    F32 xi   = SpectrumXYZLUT::color_matching_functions[4 * (i   - 360) + 1];
    F32 yi   = SpectrumXYZLUT::color_matching_functions[4 * (i   - 360) + 2];
    F32 zi   = SpectrumXYZLUT::color_matching_functions[4 * (i   - 360) + 3];
    F32 xi_1 = SpectrumXYZLUT::color_matching_functions[4 * (i+1 - 360) + 1];
    F32 yi_1 = SpectrumXYZLUT::color_matching_functions[4 * (i+1 - 360) + 2];
    F32 zi_1 = SpectrumXYZLUT::color_matching_functions[4 * (i+1 - 360) + 3];
    den += (yi + yi_1)*0.5f;
    if (min_wavelength > i) { continue; }
    if (max_wavelength < i) { continue; }
    xyz += Vec3{0.5f * (xi + xi_1), 0.5f * (yi + yi_1), 0.5f * (zi + zi_1) };
  }
  xyz /= den;
  return ColorXYZ{ xyz.x ,xyz.y ,xyz.z };
}

auto hikari::spectrum::SpectrumUniformObject::getPropertyNames() const -> std::vector<Str>
{
  return { "min_wavelength","max_wavelength","intensity","color_setting","color.xyz" ,"color.rgb" };
}

void hikari::spectrum::SpectrumUniformObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const {
  pb.clear();
  pb.setValue("max_wavelength", getMaxWaveLength());
  pb.setValue("min_wavelength", getMinWaveLength());
  pb.setValue("intensity", getIntensity());
  pb.setValue("color_setting", ColorSetting(getColorSetting()));
}

void hikari::spectrum::SpectrumUniformObject::setPropertyBlock(const PropertyBlockBase<Object>& pb) {
  auto min_wavelength = pb.getValue<F32>("min_wavelength");
  auto max_wavelength = pb.getValue<F32>("max_wavelength");
  auto intensity      = pb.getValue<F32>("intensity");
  auto color_setting  = pb.getValue("color_setting").getValue<ColorSetting>();
  if (min_wavelength) { setMinWaveLength(*min_wavelength); }
  if (max_wavelength) { setMaxWaveLength(*max_wavelength); }
  if (intensity) { setIntensity(*intensity); }
  if (color_setting) { setColorSetting(color_setting.getObject()); }
}

bool hikari::spectrum::SpectrumUniformObject::hasProperty(const Str& name) const {
  if (name == "min_wavelength") { return true; }
  if (name == "max_wavelength") { return true; }
  if (name == "intensity") { return true; }
  if (name == "color.rgb") { return true; }
  if (name == "color.xyz") { return true; }
  if (name == "color_setting") { return true; }
  return false;
}

bool hikari::spectrum::SpectrumUniformObject::getProperty(const Str& name, PropertyBase<Object>& prop) const {
  if (name == "min_wavelength") { prop = getMinWaveLength(); return true; }
  if (name == "max_wavelength") { prop = getMaxWaveLength(); return true; }
  if (name == "intensity") { prop = getIntensity(); return true; }
  if (name == "color.xyz") { auto xyz = getXYZColor(); prop = Vec3(xyz.x, xyz.y, xyz.z); return true; }
  if (name == "color_setting") { auto color_setting = getColorSetting(); prop = ColorSetting(color_setting); return true; }
  return false;
}

bool hikari::spectrum::SpectrumUniformObject::setProperty(const Str& name, const PropertyBase<Object>& prop) {
  if (name == "min_wavelength") { auto value = prop.getValue<F32>(); if (value) { setMinWaveLength(*value); return true; } return false; }
  if (name == "max_wavelength") { auto value = prop.getValue<F32>(); if (value) { setMaxWaveLength(*value); return true; } return false; }
  if (name == "intensity") { auto value = prop.getValue<F32>(); if (value) { setIntensity(*value); return true; } return false; }
  if (name == "color_setting") { auto color_setting = prop.getValue<ColorSetting>(); setColorSetting(color_setting.getObject()); }
  return false;
}

auto hikari::spectrum::SpectrumUniformSerializer::getTypeString() const noexcept -> Str
{
  return spectrum::SpectrumUniformObject::TypeString();
}

auto hikari::spectrum::SpectrumUniformSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto uniform = ObjectUtils::convert<SpectrumUniformObject>(object);
  if (!uniform) { return Json(nullptr); }
  Json json;
  json["type"] = "SpectrumUniform";
  json["properties"] = Json();
  json["properties"]["intensity"     ] = uniform->getIntensity();
  json["properties"]["min_wavelength"] = uniform->getMinWaveLength();
  json["properties"]["max_wavelength"] = uniform->getMaxWaveLength();
  return json;
}

auto hikari::spectrum::SpectrumUniformDeserializer::getTypeString() const noexcept -> Str
{
  return spectrum::SpectrumUniformObject::TypeString();
}

auto hikari::spectrum::SpectrumUniformDeserializer::eval(const Json& json) const -> std::shared_ptr<Object>
{
  auto properties = json.find("properties");
  if (properties == json.end()) { return nullptr; }
  auto uniform = SpectrumUniformObject::create();
  if (properties.value().is_null()) { return uniform; }
  if (!properties.value().is_object()) { return nullptr; }
  auto min_wavelength = properties.value().find("min_wavelength");
  auto max_wavelength = properties.value().find("max_wavelength");
  auto intensity = properties.value().find("intensity");
  if (max_wavelength != properties.value().end()) {
    try {
      auto val = max_wavelength.value().get<F32>();
      uniform->setMaxWaveLength(val);
    }
    catch (...) {
      return nullptr;
    }
  }
  if (min_wavelength != properties.value().end()) {
    try {
      auto val = min_wavelength.value().get<F32>();
      uniform->setMinWaveLength(val);
    }
    catch (...) {
      return nullptr;
    }
  }
  if (intensity != properties.value().end()) {
    try {
      auto val = intensity.value().get<F32>();
      uniform->setIntensity(val);
    }
    catch (...) {
      return nullptr;
    }
  }
  return uniform;
}
