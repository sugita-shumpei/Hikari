#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // RGB色空間で色を指定する
    // RGB色空間はコンテキストで指定したものを使用する
    struct ColorRGB  {
      float r;
      float g;
      float b;
    };
    struct ColorRGBA {
      float r;
      float g;
      float b;
      float a;
    };
    // HSB色空間で色を指定する(ただしHは(0~1)に正規化されている点に注意)
    // 描画処理時には, RGB色空間へ変換してから処理を行う
    struct ColorHSV  {
      float h;
      float s;
      float v;
    };
    // CIE1931色空間のXYZとして色を指定する
    // 描画処理時には, RGB色空間へ変換してから処理を行う
    struct ColorXYZ  {
      float x;
      float y;
      float z;
    };
    struct ColorXYZA {
      float x;
      float y;
      float z;
      float a;
    };
    enum class ColorSpace {
      eDefault,
      eCIE1931,
      eSRGB,
      eAdobeRGB,
      eRec709,
      eRec2020
    };
    // 線形RGBから非線形RGBへ変換する
    ColorRGB convertLinearRGB2NonLinearRGB(const ColorRGB& rgb, ColorSpace colorSpace = ColorSpace::eSRGB);
    // 非線形RGBから線形RGBへ変換する
    ColorRGB convertNonLinearRGB2LinearRGB(const ColorRGB& rgb, ColorSpace colorSpace = ColorSpace::eSRGB);
    // HSVから線形RGBへ変換する
    ColorHSV convertRGB2HSV (const ColorRGB& rgb);
    // RGBから線形HSBへ変換する
    ColorRGB convertHSV2RGB (const ColorHSV& hsv);
    // 線形RGBの色空間変換
    ColorRGB convertRGB2RGB (const ColorRGB& rgb, ColorSpace fromColorSpace, ColorSpace toColorSpace);
    // 線形RGBからXYZへ変換する
    ColorXYZ convertRGB2XYZ (const ColorRGB& rgb, ColorSpace colorSpace = ColorSpace::eCIE1931);
    // XYZから線形RGBへ変換する
    ColorRGB convertXYZ2RGB (const ColorXYZ& rgb, ColorSpace colorSpace = ColorSpace::eCIE1931);
    // AdobeRGB: XYZ→NormalizeXYZ
    ColorXYZ convertXYZ2NormalizeXYZ_AdobeRGB(const ColorXYZ& xyz);
    // AdobeRGB: NormalizeXYZ→XYZ
    ColorXYZ convertNormalizeXYZ2XYZ_AdobeRGB(const ColorXYZ& xyz);
    // 
    inline auto convertColorSpace2Str(const ColorSpace& color_space) -> Str {
      if (color_space == ColorSpace::eDefault ) { return "Default"; }
      if (color_space == ColorSpace::eCIE1931 ) { return "CIE1931"; }
      if (color_space == ColorSpace::eSRGB    ) { return "SRGB"; }
      if (color_space == ColorSpace::eAdobeRGB) { return "AdobeRGB"; }
      if (color_space == ColorSpace::eRec709  ) { return "Rec709"; }
      if (color_space == ColorSpace::eRec2020 ) { return "Rec2020"; }
      return "Default";
    }
    inline auto convertStr2ColorSpace(const Str& str) -> std::optional<ColorSpace> {
      if (str == "Default" ) { return ColorSpace::eDefault; }
      if (str == "CIE1931" ) { return ColorSpace::eCIE1931; }
      if (str == "SRGB"    ) { return ColorSpace::eSRGB; }
      if (str == "AdobeRGB") { return ColorSpace::eAdobeRGB; }
      if (str == "Rec709"  ) { return ColorSpace::eRec709; }
      if (str == "Rec2020" ) { return ColorSpace::eRec2020; }
      return std::nullopt;
    }
    // ColorSetting
    struct ColorSettingObject : public Object {
      using base_type = Object;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "ColorSetting"; };
      static auto create(ColorSpace color_space = ColorSpace::eCIE1931) -> std::shared_ptr<ColorSettingObject> {
        return std::shared_ptr<ColorSettingObject>(new ColorSettingObject(color_space));
      }
      virtual ~ColorSettingObject() noexcept {}
      Str getTypeString() const noexcept override { return TypeString(); }
      Bool isConvertible(const Str& type) const noexcept override { return Convertible(type); }
      auto getPropertyNames() const->std::vector<Str> override
      {
        return std::vector<Str>{ "default_colorspace" };
      }
      void getPropertyBlock(PropertyBlockBase<Object>& pb) const override
      {
        pb.clear();
        pb.setValue("default_colorspace", convertColorSpace2Str(getDefaultColorSpace()));
      }
      void setPropertyBlock(const PropertyBlockBase<Object>& pb) override
      {
        auto color_space_str = pb.getValue("default_colorspace").getValue<Str>();
        if (color_space_str) {
          auto color_space = convertStr2ColorSpace(*color_space_str);
          if (color_space) {
            setDefaultColorSpace(*color_space);
          }
        }
      }
      Bool hasProperty(const Str& name) const override
      {
        if (name == "default_colorspace") { return true; }
        return Bool();
      }
      Bool getProperty(const Str& name, PropertyBase<Object>& prop) const override
      {
        if (name == "default_colorspace") { prop = convertColorSpace2Str(getDefaultColorSpace()); return true; }
        return Bool();
      }
      Bool setProperty(const Str& name, const PropertyBase<Object>& prop) override
      {
        if (name == "default_colorspace") {
          auto color_space_str = prop.getValue<Str>();
          if (!color_space_str) { return false; }

          auto color_space = convertStr2ColorSpace(*color_space_str);
          if (!color_space) { return false; }

          setDefaultColorSpace(*color_space);
          return true;
        }
        return Bool();
      }
      void setDefaultColorSpace(ColorSpace color_space) { m_default_colorspace = color_space; }
      auto getDefaultColorSpace()const -> ColorSpace { return m_default_colorspace; }
    private:
      ColorSettingObject(ColorSpace color_space) noexcept :Object(), m_default_colorspace{ color_space } {
        if (color_space == ColorSpace::eDefault) { m_default_colorspace = ColorSpace::eCIE1931; }
      }
      ColorSpace m_default_colorspace = ColorSpace::eCIE1931;
    };
    struct ColorSetting : protected ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, ColorSettingObject> {
      using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, ColorSettingObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      ColorSetting() noexcept : impl_type() {}
      ColorSetting(const ColorSpace& color_space) :impl_type{ ColorSettingObject::create(color_space) } {}
      ColorSetting(nullptr_t) noexcept : impl_type(nullptr) {}
      ColorSetting(const std::shared_ptr<ColorSettingObject>& object) noexcept : impl_type(object) {}
      ColorSetting(const ColorSetting& opb) noexcept : impl_type(opb.getObject()) {}
      ColorSetting(ColorSetting&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      ColorSetting& operator=(const ColorSetting& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      ColorSetting& operator=(ColorSetting&& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
          opb.setObject({});
        }
        return *this;
      }
      ColorSetting& operator=(const std::shared_ptr<ColorSettingObject>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<ColorSettingObject, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      ColorSetting(const ObjectWrapperLike& wrapper) noexcept : impl_type(wrapper.getObject()) {}
      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<ColorSettingObject, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      ColorSetting& operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      void setDefaultColorSpace(ColorSpace color_space) { auto object = getObject(); if (object) { object->setDefaultColorSpace(color_space); } }
      auto getDefaultColorSpace()const -> ColorSpace {
        auto object = getObject(); if (!object) { return ColorSpace::eCIE1931; }
        else { return object->getDefaultColorSpace(); }
      }

      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::operator[];
      using impl_type::isConvertible;
      using impl_type::getName;
      using impl_type::getPropertyNames;
      using impl_type::getObject;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getValue;
      using impl_type::hasValue;
      using impl_type::setValue;
    };
  }
}
