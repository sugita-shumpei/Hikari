#include <hikari/spectrum/color.h>
auto hikari::spectrum::SpectrumColorObject::getPropertyNames() const -> std::vector<Str>
{
  return { "min_wavelength","max_wavelength","color.rgb","color.xyz","color.rgb.value","color.rgb.color_space","color.rgb.linear", "color_setting" };
}

void hikari::spectrum::SpectrumColorObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const
{
  pb.clear();
  pb.setValue("max_wavelength", getMaxWaveLength());
  pb.setValue("min_wavelength", getMinWaveLength());
  ColorSpace color_space = ColorSpace::eDefault; ColorRGB rgb = {}; Bool linear = true;
  if (getRGBColor(rgb, color_space, linear)) {
    pb.setValue("color.rgb.value", Vec3(rgb.r, rgb.g, rgb.b));
    pb.setValue("color.rgb.color_space", convertColorSpace2Str(color_space));
    pb.setValue("color.rgb.linear", linear);
  }
  else {
    auto p_xyz_color = std::get_if<1>(&m_handle);
    pb.setValue("color.xyz", Vec3(p_xyz_color->x, p_xyz_color->y, p_xyz_color->z));
  }
  pb.setValue("color_setting", ColorSetting(getColorSetting()));
}

void hikari::spectrum::SpectrumColorObject::setPropertyBlock(const PropertyBlockBase<Object>& pb)
{
  auto rgb_color = pb.getValue("color.rgb").toVec();
  if (!rgb_color) {
    rgb_color = pb.getValue("color.rgb.value").toVec();
    if (rgb_color) {
      auto p_rgb_color_space = pb.getValue("color.rgb.color_space").getValue<Str>();
      auto pp_rgb_linear = pb.getValue("color.rgb.linear");
      auto p_rgb_linear = pp_rgb_linear.getValue<Bool>();
      auto rgb_color_space = ColorSpace::eDefault;
      bool rgb_linear = true;
      if (p_rgb_color_space) {
        auto tmp = convertStr2ColorSpace(*p_rgb_color_space);
        if (tmp) { rgb_color_space = *tmp; }
      }
      if (p_rgb_linear) {
        rgb_linear = *p_rgb_linear;
      }
      setRGBColor(ColorRGB{ rgb_color->x,rgb_color->y,rgb_color->z }, rgb_color_space, rgb_linear);
    }
  }
  else {
    setRGBColor(ColorRGB{ rgb_color->x,rgb_color->y,rgb_color->z }, ColorSpace::eDefault, true);
  }
  auto xyz_color = pb.getValue("color.xyz").toVec();
  if (xyz_color) {
    setXYZColor(ColorXYZ{ xyz_color->x,xyz_color->y,xyz_color->z });
  }
  auto color_setting = pb.getValue("color_setting").getValue<ColorSetting>();
  if (color_setting) { setColorSetting(color_setting.getObject()); }
}

bool  hikari::spectrum::SpectrumColorObject::hasProperty(const Str& name) const
{
  if (name == "min_wavelength") { return true; }
  if (name == "max_wavelength") { return true; }
  if (name == "color_setting") { return true; }
  if (name == "color.rgb") { return true; }
  if (name == "color.xyz") { return true; }
  if (m_handle.index() == 0) {
    if (name == "color.rgb.value") { return true; }
    if (name == "color.rgb.color_space") { return true; }
    if (name == "color.rgb.linear") { return true; }
  }
  return false;
}

bool  hikari::spectrum::SpectrumColorObject::getProperty(const Str& name, PropertyBase<Object>& prop) const
{
  if (name == "min_wavelength") { prop.setValue(getMinWaveLength()); return true; }
  if (name == "max_wavelength") { prop.setValue(getMaxWaveLength()); return true; }
  if (name == "color.xyz") { auto color = getXYZColor(); prop.setValue(Vec3(color.x, color.y, color.z)); return true; }
  if (name == "color.rgb") { auto color = getRGBColor(ColorSpace::eDefault, true); prop.setValue(Vec3(color.r, color.g, color.b)); return true; }
  if (name == "color.rgb.value") {
    ColorSpace color_space = ColorSpace::eDefault; ColorRGB rgb = {}; Bool linear = true;
    if (getRGBColor(rgb, color_space, linear)) {
      prop.setValue(Vec3(rgb.r, rgb.g, rgb.b));
    }
    else {
      auto color = getRGBColor(ColorSpace::eDefault, true);
      prop.setValue(Vec3(color.r, color.g, color.b));
    }
    return true;
  }
  if (name == "color.rgb.color_space") {
    auto color_space = getColorSpace();
    prop.setValue(convertColorSpace2Str(color_space));
    return true;
  }
  if (name == "color.rgb.linear") {
    auto islinear = isLinear();
    prop.setValue(islinear);
    return true;
  }
  if (name == "color_setting") { auto color_setting = getColorSetting(); prop = ColorSetting(color_setting); return true; }
  return false;
}

bool  hikari::spectrum::SpectrumColorObject::setProperty(const Str& name, const PropertyBase<Object>& prop)
{
  if (name == "color.xyz") { auto color = prop.toVec(); if (!color) { return false; }; setXYZColor(ColorXYZ{ color->x,color->y,color->z }); return true; }
  if (name == "color.rgb") { auto color = prop.toVec(); if (!color) { return false; }; setRGBColor(ColorRGB{ color->x,color->y,color->z }); return true; }
  if (name == "color_setting") { auto color_setting = prop.getValue<ColorSetting>(); setColorSetting(color_setting.getObject()); return true; }
  return false;
}

auto hikari::spectrum::SpectrumColorObject::sample(F32 wavelength) const -> F32
{
  auto xyz_color = getXYZColor();

  return F32();
}

auto hikari::spectrum::SpectrumColorObject::getRGBColor(ColorSpace to_color_space, bool  is_linear) const -> ColorRGB {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (!p_rgb_handle) {
    // XYZ Color
    if (to_color_space == ColorSpace::eDefault) {
      auto setting = ColorSetting(getColorSetting());
      to_color_space = setting.getDefaultColorSpace();
    }
    auto linear_rgb = convertXYZ2RGB(std::get<1>(m_handle), to_color_space);
    if (is_linear) { return linear_rgb; }
    return convertLinearRGB2NonLinearRGB(linear_rgb, to_color_space);
  }
  else {
    auto from_color_space = p_rgb_handle->colorSpace;
    if (from_color_space == ColorSpace::eDefault) {
      auto setting = ColorSetting(getColorSetting());
      from_color_space = setting.getDefaultColorSpace();
    }
    if (to_color_space == ColorSpace::eDefault) {
      auto setting = ColorSetting(getColorSetting());
      to_color_space = setting.getDefaultColorSpace();
    }
    auto src_rgb = p_rgb_handle->rgb;
    if (from_color_space == to_color_space) {
      if (is_linear == p_rgb_handle->isLinear) { return src_rgb; }
      else {
        if (is_linear) { return convertNonLinearRGB2LinearRGB(src_rgb, to_color_space); }
        else { return convertLinearRGB2NonLinearRGB(src_rgb, to_color_space); }
      }
    }
    if (!p_rgb_handle->isLinear) { src_rgb = convertNonLinearRGB2LinearRGB(src_rgb, from_color_space); }
    auto dst_rgb = convertRGB2RGB(src_rgb, from_color_space, to_color_space);
    if (!is_linear) { dst_rgb = convertLinearRGB2NonLinearRGB(dst_rgb, to_color_space); }
    return dst_rgb;
  }
}
auto hikari::spectrum::SpectrumColorObject::getXYZColor() const -> ColorXYZ {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (!p_rgb_handle) {
    return std::get<1>(m_handle);
  }
  else {
    auto from_color_space = p_rgb_handle->colorSpace;
    if (from_color_space == ColorSpace::eDefault) {
      auto setting = ColorSetting(getColorSetting());
      from_color_space = setting.getDefaultColorSpace();
    }
    auto src_rgb = p_rgb_handle->rgb;
    if (!p_rgb_handle->isLinear) { src_rgb = convertNonLinearRGB2LinearRGB(src_rgb, from_color_space); }
    return convertRGB2XYZ(src_rgb, from_color_space);
  }
}

auto hikari::spectrum::SpectrumColorObject::getMinWaveLength() const -> F32 { return 360.0f; }

auto hikari::spectrum::SpectrumColorObject::getMaxWaveLength() const -> F32 { return 830.0f; }

bool hikari::spectrum::SpectrumColorObject::getRGBColor(ColorRGB& rgb, ColorSpace& color_space, Bool& is_linear) const
{
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (!p_rgb_handle) {
    return false;
  }
  else {
    rgb = p_rgb_handle->rgb;
    color_space = p_rgb_handle->colorSpace;
    is_linear = p_rgb_handle->isLinear;
    return true;
  }
}

void hikari::spectrum::SpectrumColorObject::setRGBColor(const ColorRGB& rgb, ColorSpace color_space, Bool  is_linear) {
  m_handle = RGBHandle{ rgb,color_space,is_linear };
}

void hikari::spectrum::SpectrumColorObject::setXYZColor(const ColorXYZ& xyz) { m_handle = xyz; }

auto hikari::spectrum::SpectrumColorObject::getColorSpace() const -> ColorSpace {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (!p_rgb_handle) {
    return ColorSpace::eDefault;
  }
  else {
    return p_rgb_handle->colorSpace;
  }
}

void hikari::spectrum::SpectrumColorObject::setColorSpace(const ColorSpace& color_space) {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (p_rgb_handle) { p_rgb_handle->colorSpace = color_space; }
}

void hikari::spectrum::SpectrumColorObject::setLinear(Bool  is_linear) {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (p_rgb_handle) { p_rgb_handle->isLinear = is_linear; }
}

bool  hikari::spectrum::SpectrumColorObject::isLinear() const noexcept {
  auto p_rgb_handle = std::get_if<0>(&m_handle);
  if (!p_rgb_handle) {
    return true;
  }
  else {
    return p_rgb_handle->isLinear;
  }
}
auto hikari::spectrum::SpectrumColorSerializer::getTypeString() const noexcept -> Str
{
  return SpectrumColorObject::TypeString();
}

auto hikari::spectrum::SpectrumColorSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto color = ObjectUtils::convert<SpectrumColorObject>(object);
  if (!color) { return Json(); }
  Json json          = {};
  json["type"]       = "SpectrumColor";
  json["properties"] = Json();
  json["properties"]["color"] = {};
  ColorRGB rgb = {}; ColorSpace colorspace = ColorSpace::eDefault; Bool is_linear = true;
  if (color->getRGBColor(rgb, colorspace,is_linear)) {
    json["properties"]["color"]["rgb"] = {};
    json["properties"]["color"]["rgb"]["value"]       = std::array<F32, 3>{rgb.r, rgb.g, rgb.b};
    json["properties"]["color"]["rgb"]["color_space"] = convertColorSpace2Str(colorspace);
    json["properties"]["color"]["rgb"]["is_linear"]   = is_linear;
  }
  else {
    auto xyz = color->getXYZColor();
    json["properties"]["color"]["xyz"] = {};
    json["properties"]["color"]["xyz"] = std::array<F32, 3>{xyz.x, xyz.y, xyz.z};
  }
  return json;
}

auto hikari::spectrum::SpectrumColorDeserializer::getTypeString() const noexcept -> Str
{
  return SpectrumColorObject::TypeString();
}

auto hikari::spectrum::SpectrumColorDeserializer::eval(const Json& json) const -> std::shared_ptr<Object>
{
  auto properties = json.find("properties");
  if (properties == json.end()) { return nullptr; }
  if (properties.value().is_null()) { return nullptr; }
  if (!properties.value().is_object()) { return nullptr; }
  auto color      = properties.value().find("color");
  if (color == properties.value().end()) { return nullptr; }
  if (!color.value().is_object()) { return nullptr; }
  auto rgb = color.value().find("rgb");
  auto xyz = color.value().find("xyz");
  if (rgb != color.value().end()) {
    if (!rgb.value().is_object()) { return nullptr; }
    auto value = rgb.value().find("value");
    auto color_space = rgb.value().find("color_space");
    auto is_linear = rgb.value().find("is_linear");
    auto color_rgb = ColorRGB();
    auto k_color_space = ColorSpace();
    bool k_linear = true;
    if (value != rgb.value().end()) {
      try {
        auto tmp  = value.value().get<std::array<F32, 3>>();
        color_rgb = { tmp[0],tmp[1],tmp[2] };
      }
      catch (...) {
        return nullptr;
      }
    }
    else {
      return nullptr;
    }
    if (color_space != rgb.value().end()) {
      try {
        auto tmp  = color_space.value().get<Str>();
        auto tmp2 = convertStr2ColorSpace(tmp);
        if (!tmp2) { return nullptr; }
        k_color_space = *tmp2;
      }
      catch (...) {
        return nullptr;
      }
    }
    if (is_linear != rgb.value().end()) {
      try {
        k_linear = is_linear.value().get<Bool>();
      }
      catch (...) {
        return nullptr;
      }
    }
    return SpectrumColorObject::create(color_rgb, k_color_space, k_linear);
  }
  if (xyz != color.value().end()) {
    try {
      auto tmp = xyz.value().get<std::array<F32, 3>>();
      return SpectrumColorObject::create(ColorXYZ{ tmp[0], tmp[1], tmp[2] });
    }
    catch (...) {
      return nullptr;
    }
  }
  return nullptr;
}
