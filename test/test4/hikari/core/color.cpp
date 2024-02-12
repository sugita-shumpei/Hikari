#include <hikari/core/color.h>
#include <hikari/core/data_type.h>
hikari::core::ColorHSV hikari::core::convertRGB2HSV(const ColorRGB& rgb)
{
  if ((rgb.r > 1.0f) || (rgb.g > 1.0f) || (rgb.b > 1.0f)) { return ColorHSV{ 0.0f,0.0f,0.0f }; }
  if ((rgb.r < 0.0f) || (rgb.g < 0.0f) || (rgb.b < 0.0f)) { return ColorHSV{ 0.0f,0.0f,0.0f }; }
  float max = fmaxf(fmaxf(rgb.r, rgb.g), rgb.b);
  float min = fminf(fminf(rgb.r, rgb.g), rgb.b);
  ColorHSV hsv = {};
  do {
    if (max == min) { continue; }
    if (max == rgb.r) {
      // (-1,1) -> (5~5.9999)+(0,1) ->(300~360),(0,60)
      hsv.h = (1.0f * fmodf((rgb.g - rgb.b) / (max - min),6.0f))/6.0f;
      break;
    }
    if (max == rgb.g) {
      // (-1,1) -> (60,180)
      hsv.h = ((rgb.b - rgb.r) / (max - min) + 2.0f)/6.0f;
      break;
    }
    if (max == rgb.b) {
      // (-1,1) -> (180,300)
      hsv.h = ((rgb.r - rgb.g) / (max - min) + 4.0f)/6.0f;
      break;
    }
  } while (0);
  hsv.v =  max;
  hsv.s = max > 0.0f ? (max - min) / max : 0.0f;
  return hsv;
}

hikari::core::ColorRGB hikari::core::convertHSV2RGB(const ColorHSV& hsv)
{
  if ((hsv.h > 1.0f) || (hsv.s > 1.0f) || (hsv.v > 1.0f)) { return ColorRGB{ 0.0f,0.0f,0.0f }; }
  if ((hsv.h < 0.0f) || (hsv.s < 0.0f) || (hsv.v < 0.0f)) { return ColorRGB{ 0.0f,0.0f,0.0f }; }
  if (hsv.s == 0.0f) { return ColorRGB{ hsv.v,hsv.v,hsv.v }; }
  float max_min = hsv.v * hsv.s;
  float max     = hsv.v;
  float min     = fmaxf(max - max_min,0.0f);
  float h       = 6.0f * hsv.h;
  if ((h  < 1.0f)) {
    float r     = max;//r=1
    float g_m_b = h;// h=1->g=1,b=0
    float b     = min;
    float g     = g_m_b + b;
    return ColorRGB{ r,g,b };
  }
  if ((h  < 2.0f)) {
    float g     = max;// g=1
    float r_m_b = 2.0f - h;// h=1->r_m_b=1->r=1,b=0
    float b     = min;
    float r     = r_m_b + b;
    return ColorRGB{ r,g,b };
  }
  if ((h  < 3.0f)) {
    float g     = max;// g=1
    float b_m_r = h - 2.0f;// h=1->r_m_b=1->r=1,b=0
    float r     = min;
    float b     = b_m_r + r;
    return ColorRGB{ r,g,b };
  }
  if ((h  < 4.0f)) {
    float b     = max;// g=1
    float g_m_r = 4.0f - h;// h=1->r_m_b=1->r=1,b=0
    float r     = min;
    float g     = g_m_r + r;
    return ColorRGB{ r,g,b };
  }
  if ((h  < 5.0f)) {
    float b     = max;// g=1
    float r_m_g = h - 4.0f;// h=1->r_m_b=1->r=1,b=0
    float g     = min;
    float r     = r_m_g + g;
    return ColorRGB{ r,g,b };
  }
  if ((h <= 6.0f)) {
    float r     = max;
    float b_m_g = 6-h;
    float g     = min;
    float b     = b_m_g + g;
    return ColorRGB{ r,g,b };
  }
  return ColorRGB{ 0.0f,0.0f,0.0f };
}

hikari::core::ColorRGB hikari::core::convertRGB2RGB(const ColorRGB& rgb, ColorSpace fromColorSpace, ColorSpace toColorSpace)
{
  if (fromColorSpace == toColorSpace) { return rgb; }
  ColorXYZ from_xyz = convertRGB2XYZ(rgb, fromColorSpace);
  return convertXYZ2RGB(from_xyz,toColorSpace);
}
// CIE1931: RGB→XYZ
static hikari::core::ColorXYZ convertRGB2XYZ_CIE1931(const hikari::core::ColorRGB& rgb) {
  hikari::Vec3 vec_xyz = static_cast<hikari::F32>(1.0 / 0.17697) * hikari::Mat3(
    hikari::Vec3(0.49f, 0.17697f, 0.0f) ,
    hikari::Vec3(0.31f, 0.81240f, 0.01f),
    hikari::Vec3(0.20f, 0.01f   , 0.99f)
  )*hikari::Vec3(rgb.r,rgb.g,rgb.b);
  return hikari::ColorXYZ{ vec_xyz.x,vec_xyz.y,vec_xyz.z };
}
// CIE1931: XYZ→RGB
static hikari::core::ColorRGB convertXYZ2RGB_CIE1931(const hikari::core::ColorXYZ& xyz) {
  hikari::Vec3 vec_rgb = hikari::Mat3(
    hikari::Vec3( 0.41847f , -0.091169f, 0.00092090f),
    hikari::Vec3(-0.15866f ,   0.25243f, -0.0025498f),
    hikari::Vec3(-0.082835f,  0.015708f,    0.17860f)
  ) * hikari::Vec3(xyz.x, xyz.y, xyz.z);
  return hikari::ColorRGB{ vec_rgb.r,vec_rgb.g,vec_rgb.b };
}
// sRGB: RGB→XYZ
static hikari::core::ColorXYZ convertRGB2XYZ_SRGB(const hikari::core::ColorRGB& rgb) {
  hikari::Vec3 vec_xyz = hikari::Mat3(
    hikari::Vec3(0.412391f, 0.212639f, 0.0193308f),
    hikari::Vec3(0.357584f, 0.715169f, 0.119195f),
    hikari::Vec3(0.180481f, 0.0721923f, 0.950532f)
  ) * hikari::Vec3(rgb.r, rgb.g, rgb.b);
  return hikari::ColorXYZ{ vec_xyz.x,vec_xyz.y,vec_xyz.z };
}
// sRGB: XYZ→RGB
static hikari::core::ColorRGB convertXYZ2RGB_SRGB(const hikari::core::ColorXYZ& xyz) {
  hikari::Vec3 vec_rgb = hikari::Mat3(
    hikari::Vec3(3.24097f, -0.969244f, 0.0556301f),
    hikari::Vec3(-1.53738f, 1.87597f, -0.203977f),
    hikari::Vec3(-0.498611f, 0.0415551f, 1.05697f)
  ) * hikari::Vec3(xyz.x, xyz.y, xyz.z);
  return hikari::ColorRGB{ vec_rgb.r,vec_rgb.g,vec_rgb.b };
}
// AdobeRGB: RGB→XYZ
static hikari::core::ColorXYZ convertRGB2XYZ_AdobeRGB(const hikari::core::ColorRGB& rgb) {
  hikari::Vec3 vec_xyz = hikari::Mat3(
    hikari::Vec3(+0.57667f, +0.29734f, +0.02703f),
    hikari::Vec3(+0.18556f, +0.62736f, +0.07069f),
    hikari::Vec3(+0.18823f, +0.07529f, +0.99134f)
  ) * hikari::Vec3(rgb.r, rgb.g, rgb.b);
  return hikari::ColorXYZ{ vec_xyz.x,vec_xyz.y,vec_xyz.z };
}
// AdobeRGB: XYZ→RGB
static hikari::core::ColorRGB convertXYZ2RGB_AdobeRGB(const hikari::core::ColorXYZ& xyz) {
  hikari::Vec3 vec_rgb = hikari::Mat3(
    hikari::Vec3(+2.04159f,-0.96924f,+0.01344f),
    hikari::Vec3(-0.56501f,+1.87597f,-0.11836f),
    hikari::Vec3(-0.34473f,+0.04156f,+1.01517f)
  ) * hikari::Vec3(xyz.x, xyz.y, xyz.z);
  return hikari::ColorRGB{ vec_rgb.r,vec_rgb.g,vec_rgb.b };
}
// Rec.709: RGB→XYZ(sRGBと同一)
static hikari::core::ColorXYZ convertRGB2XYZ_Rec709(const hikari::core::ColorRGB& rgb) {
  return convertRGB2XYZ_SRGB(rgb);
}
// Rec.709: XYZ→RGB(sRGBと同一)
static hikari::core::ColorRGB convertXYZ2RGB_Rec709(const hikari::core::ColorXYZ& xyz) {
  return convertXYZ2RGB_SRGB(xyz);
}
// Rec.2020: RGB→XYZ
static hikari::core::ColorXYZ convertRGB2XYZ_Rec2020(const hikari::core::ColorRGB& rgb) {
  hikari::Vec3 vec_xyz = hikari::Mat3(
    hikari::Vec3(0.636958, 0.2627  , 0.0f),
    hikari::Vec3(0.144617, 0.677998, 0.0280727),
    hikari::Vec3(0.168881, 0.0593017, 1.06099)
  ) * hikari::Vec3(rgb.r, rgb.g, rgb.b);
  return hikari::ColorXYZ{ vec_xyz.x,vec_xyz.y,vec_xyz.z };
}
// Rec.2020: XYZ→RGB
static hikari::core::ColorRGB convertXYZ2RGB_Rec2020(const hikari::core::ColorXYZ& xyz) {
  hikari::Vec3 vec_rgb = hikari::Mat3(
    hikari::Vec3(1.71665, -0.666684, 0.0176399),
    hikari::Vec3(-0.355671, 1.61648, -0.0427706),
    hikari::Vec3(-0.253366, 0.0157685, 0.942103)
  ) * hikari::Vec3(xyz.x, xyz.y, xyz.z);
  return hikari::ColorRGB{ vec_rgb.r,vec_rgb.g,vec_rgb.b };
}
// sRGB
static hikari::core::F32 convertNonLinear2Linear_SRGB(const hikari::core::F32& v) {
  if (v <= 0.04045f) {
    return v * static_cast<hikari::F32>(1.0/ 12.92);
  }
  else {
    return powf(v * static_cast<hikari::F32>(1.0/1.055)+ static_cast<hikari::F32>(0.055 / 1.055), 2.4f);
  }
}
static hikari::core::F32 convertLinear2NonLinear_SRGB(const hikari::core::F32& v) {
  if (v <= 0.0031308f) {
    return 12.92f * v;
  }
  else {
    return 1.055f*powf(v, static_cast<hikari::core::F32>(1.0/2.4)) - 0.055f;
  }
}
static hikari::core::F32 convertNonLinear2Linear_AdobeRGB(const hikari::core::F32& v) {
  if (v <= 0.05557664) {
    return 32.0f * v;
  }
  else {
    return powf(v, static_cast<hikari::core::F32>(2.19921875));
  }
}
static hikari::core::F32 convertLinear2NonLinear_AdobeRGB(const hikari::core::F32& v) {
  if (v <= 0.00173677) {
    return 32.0f * v;// 0.05557664
  }
  else {
    return powf(v, static_cast<hikari::core::F32>(1.0 / 2.19921875)); //0.055576565768339 
  }
}
static hikari::core::F32 convertNonLinear2Linear_Rec709(const hikari::core::F32& v) {
  if (v <= 0.081f) {
    return v * static_cast<float>(1.0/4.5);
  }
  else {
    return powf(v * static_cast<float>(1.0 / 1.099)+ static_cast<float>(0.099 / 1.099), static_cast<float>(1.0 / 0.45));
  }
}
static hikari::core::F32 convertLinear2NonLinear_Rec709(const hikari::core::F32& v) {
  if (v <= 0.018f) {
    return 4.5f*v;
  }
  else {
    return 1.099f * powf(v, 0.45f) - 0.099f;
  }
}
static hikari::core::F32 convertNonLinear2Linear_Rec2020(const hikari::core::F32& v) {
  return convertNonLinear2Linear_Rec709(v);
}
static hikari::core::F32 convertLinear2NonLinear_Rec2020(const hikari::core::F32& v) {
  return convertLinear2NonLinear_Rec709(v);
}
// sRGB: NonLinear→Linear
static hikari::core::ColorRGB convertLinearRGB2NonLinearRGB_SRGB(const hikari::core::ColorRGB& linear) {
  float r = convertLinear2NonLinear_SRGB(linear.r);
  float g = convertLinear2NonLinear_SRGB(linear.g);
  float b = convertLinear2NonLinear_SRGB(linear.b);
  return hikari::core::ColorRGB{r, g, b};
}
static hikari::core::ColorRGB convertNonLinearRGB2LinearRGB_SRGB(const hikari::core::ColorRGB& nonlinear) {
  float r = convertNonLinear2Linear_SRGB(nonlinear.r);
  float g = convertNonLinear2Linear_SRGB(nonlinear.g);
  float b = convertNonLinear2Linear_SRGB(nonlinear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
// AdobeRGB: NonLinear→Linear
static hikari::core::ColorRGB convertLinearRGB2NonLinearRGB_AdobeRGB(const hikari::core::ColorRGB& linear) {
  float r = convertLinear2NonLinear_AdobeRGB(linear.r);
  float g = convertLinear2NonLinear_AdobeRGB(linear.g);
  float b = convertLinear2NonLinear_AdobeRGB(linear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
static hikari::core::ColorRGB convertNonLinearRGB2LinearRGB_AdobeRGB(const hikari::core::ColorRGB& nonlinear) {
  float r = convertNonLinear2Linear_AdobeRGB(nonlinear.r);
  float g = convertNonLinear2Linear_AdobeRGB(nonlinear.g);
  float b = convertNonLinear2Linear_AdobeRGB(nonlinear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
// Rec.709: NonLinear→Linear
static hikari::core::ColorRGB convertLinearRGB2NonLinearRGB_Rec709(const hikari::core::ColorRGB& linear) {
  float r = convertLinear2NonLinear_Rec709(linear.r);
  float g = convertLinear2NonLinear_Rec709(linear.g);
  float b = convertLinear2NonLinear_Rec709(linear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
static hikari::core::ColorRGB convertNonLinearRGB2LinearRGB_Rec709(const hikari::core::ColorRGB& nonlinear) {
  float r = convertNonLinear2Linear_Rec709(nonlinear.r);
  float g = convertNonLinear2Linear_Rec709(nonlinear.g);
  float b = convertNonLinear2Linear_Rec709(nonlinear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
// Rec.2020: NonLinear→Linear
static hikari::core::ColorRGB convertLinearRGB2NonLinearRGB_Rec2020(const hikari::core::ColorRGB& linear) {
  float r = convertLinear2NonLinear_Rec2020(linear.r);
  float g = convertLinear2NonLinear_Rec2020(linear.g);
  float b = convertLinear2NonLinear_Rec2020(linear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
static hikari::core::ColorRGB convertNonLinearRGB2LinearRGB_Rec2020(const hikari::core::ColorRGB& nonlinear) {
  float r = convertNonLinear2Linear_Rec2020(nonlinear.r);
  float g = convertNonLinear2Linear_Rec2020(nonlinear.g);
  float b = convertNonLinear2Linear_Rec2020(nonlinear.b);
  return hikari::core::ColorRGB{ r, g, b };
}
hikari::core::ColorXYZ hikari::core::convertRGB2XYZ(const ColorRGB& rgb, ColorSpace colorSpace)
{
  if (colorSpace == ColorSpace::eCIE1931)  { return convertRGB2XYZ_CIE1931(rgb); }
  if (colorSpace == ColorSpace::eSRGB)     { return convertRGB2XYZ_SRGB(rgb);    }
  if (colorSpace == ColorSpace::eAdobeRGB) { return convertRGB2XYZ_AdobeRGB(rgb);}
  if (colorSpace == ColorSpace::eRec709  ) { return convertRGB2XYZ_Rec709(rgb);  }
  if (colorSpace == ColorSpace::eRec2020 ) { return convertRGB2XYZ_Rec2020(rgb); }
  return ColorXYZ();
}
hikari::core::ColorRGB hikari::core::convertXYZ2RGB(const ColorXYZ& xyz, ColorSpace colorSpace)
{
  if (colorSpace == ColorSpace::eCIE1931)  { return convertXYZ2RGB_CIE1931(xyz); }
  if (colorSpace == ColorSpace::eSRGB)     { return convertXYZ2RGB_SRGB(xyz); }
  if (colorSpace == ColorSpace::eAdobeRGB) { return convertXYZ2RGB_AdobeRGB(xyz);}
  if (colorSpace == ColorSpace::eRec709  ) { return convertXYZ2RGB_Rec709(xyz); }
  if (colorSpace == ColorSpace::eRec2020 ) { return convertXYZ2RGB_Rec2020(xyz); }
  return ColorRGB();
}
// AdobeRGB: XYZ→NormalizeXYZ
hikari::core::ColorXYZ hikari::core::convertXYZ2NormalizeXYZ_AdobeRGB(const ColorXYZ& xyz) {
  constexpr hikari::core::ColorXYZ black{ 0.5282f,0.5557f,0.6052f };
  constexpr hikari::core::ColorXYZ white{ 152.07f,160.00f,174.25f };
  float x = (xyz.x - black.x) * static_cast<float>(static_cast<double>(white.x - black.x) * static_cast<double>(white.x) / static_cast<double>(white.y));
  float y = (xyz.y - black.y) * static_cast<float>(white.y - black.y);
  float z = (xyz.z - black.z) * static_cast<float>(static_cast<double>(white.z - black.z) * static_cast<double>(white.z) / static_cast<double>(white.y));
  return hikari::core::ColorXYZ{ x,y,z };
}
// AdobeRGB: NormalizeXYZ→XYZ
hikari::core::ColorXYZ hikari::core::convertNormalizeXYZ2XYZ_AdobeRGB(const ColorXYZ& xyz) {
  constexpr hikari::core::ColorXYZ black{ 0.5282f,0.5557f,0.6052f };
  constexpr hikari::core::ColorXYZ white{ 152.07f,160.00f,174.25f };
  float x = xyz.x * static_cast<float>(static_cast<double>(white.y) / (static_cast<double>(white.x) * static_cast<double>(white.x - black.x))) + black.x;
  float y = xyz.y * static_cast<float>(1.0 / static_cast<double>(white.y - black.y)) + black.y;
  float z = xyz.z * static_cast<float>(static_cast<double>(white.y) / (static_cast<double>(white.z) * static_cast<double>(white.z - black.z))) + black.z;
  return hikari::core::ColorXYZ{ x,y,z };
}

hikari::core::ColorRGB hikari::core::convertNonLinearRGB2LinearRGB(const ColorRGB& rgb, ColorSpace colorSpace)
{
  if (colorSpace == ColorSpace::eCIE1931) { return rgb; }
  if (colorSpace == ColorSpace::eSRGB) { return convertNonLinearRGB2LinearRGB_SRGB(rgb); }
  if (colorSpace == ColorSpace::eAdobeRGB) { return convertNonLinearRGB2LinearRGB_AdobeRGB(rgb); }
  if (colorSpace == ColorSpace::eRec709) { return convertNonLinearRGB2LinearRGB_Rec709(rgb); }
  if (colorSpace == ColorSpace::eRec2020) { return convertNonLinearRGB2LinearRGB_Rec2020(rgb); }
  return {};
}
hikari::core::ColorRGB hikari::core::convertLinearRGB2NonLinearRGB(const ColorRGB& rgb, ColorSpace colorSpace)
{
  if (colorSpace == ColorSpace::eCIE1931) { return rgb; }
  if (colorSpace == ColorSpace::eSRGB)     { return convertLinearRGB2NonLinearRGB_SRGB(rgb); }
  if (colorSpace == ColorSpace::eAdobeRGB) { return convertLinearRGB2NonLinearRGB_AdobeRGB(rgb); }
  if (colorSpace == ColorSpace::eRec709)   { return convertLinearRGB2NonLinearRGB_Rec709(rgb); }
  if (colorSpace == ColorSpace::eRec2020)  { return convertLinearRGB2NonLinearRGB_Rec2020(rgb); }
  return {};
}

