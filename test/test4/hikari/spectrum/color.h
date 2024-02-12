#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace spectrum {
    struct SpectrumColorObject : public SpectrumObject {
      using base_type = SpectrumObject;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "SpectrumColor"; };
      static auto create(const ColorRGB& rgb, ColorSpace color_space = ColorSpace::eDefault, Bool is_linear = true) -> std::shared_ptr<SpectrumColorObject> {
        return std::shared_ptr<SpectrumColorObject>(new SpectrumColorObject(rgb, color_space, is_linear));
      }
      static auto create(const ColorXYZ& xyz) -> std::shared_ptr<SpectrumColorObject> {
        return std::shared_ptr<SpectrumColorObject>(new SpectrumColorObject(xyz));
      }
      virtual ~SpectrumColorObject() noexcept {}
      virtual auto getName() const -> Str { return ""; }
      virtual Str  getTypeString() const noexcept override { return TypeString(); }
      virtual Bool isConvertible(const Str& type) const noexcept override { return Convertible(type); }
      virtual auto getPropertyNames() const->std::vector<Str> override;
      virtual void getPropertyBlock(PropertyBlockBase<Object>& pb) const override;
      virtual void setPropertyBlock(const PropertyBlockBase<Object>& pb) override;
      virtual Bool hasProperty(const Str& name) const override;
      virtual Bool getProperty(const Str& name, PropertyBase<Object>& prop) const override;
      virtual Bool setProperty(const Str& name, const PropertyBase<Object>& prop) override;
      virtual auto sample(F32 wavelength) const->F32 override;
      virtual auto getRGBColor(ColorSpace to_color_space, Bool is_linear) const->ColorRGB override;
      virtual auto getXYZColor() const->ColorXYZ override;
      virtual auto getMinWaveLength() const->F32 override;
      virtual auto getMaxWaveLength() const->F32 override;
      bool getRGBColor(ColorRGB& rgb, ColorSpace& color_space, Bool& is_linear) const;
      void setRGBColor(const ColorRGB& rgb, ColorSpace color_space = ColorSpace::eDefault, Bool is_linear = true);
      void setXYZColor(const ColorXYZ& xyz);
      auto getColorSpace() const->ColorSpace;
      void setColorSpace(const ColorSpace& color_space);
      void setLinear(Bool is_linear);
      Bool isLinear()const noexcept;
    private:
      SpectrumColorObject(const ColorRGB& rgb, ColorSpace color_space, Bool is_linear)noexcept : SpectrumObject(), m_handle{ RGBHandle{rgb,color_space,is_linear} } {

      }
      SpectrumColorObject(const ColorXYZ& xyz)noexcept : SpectrumObject(), m_handle{ xyz } {}
    private:
      struct RGBHandle { ColorRGB rgb; ColorSpace colorSpace; Bool isLinear; };
      using  XYZHandle = ColorXYZ;
      std::variant<RGBHandle, XYZHandle> m_handle;
    };
    struct SpectrumColor : protected SpectrumImpl<SpectrumColorObject> {
      using impl_type = SpectrumImpl<SpectrumColorObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      SpectrumColor() noexcept : impl_type() {}
      SpectrumColor(const ColorRGB& rgb, ColorSpace color_space = ColorSpace::eDefault, Bool is_linear = true) noexcept : impl_type{ SpectrumColorObject::create(rgb,color_space,is_linear) } {}
      SpectrumColor(const ColorXYZ& xyz) noexcept : impl_type{ SpectrumColorObject::create(xyz) } {}
      SpectrumColor(nullptr_t) noexcept : impl_type(nullptr) {}
      SpectrumColor(const std::shared_ptr<SpectrumColorObject>& object) noexcept : impl_type(object) {}
      SpectrumColor(const SpectrumColor& opb) noexcept : impl_type(opb.getObject()) {}
      SpectrumColor(SpectrumColor&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      SpectrumColor& operator=(const SpectrumColor& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      SpectrumColor& operator=(SpectrumColor&& opb) noexcept
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
      SpectrumColor& operator=(const std::shared_ptr<SpectrumColorObject>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      bool getRGBColor(ColorRGB& rgb, ColorSpace& color_space, Bool& is_linear) const {
        auto object = getObject();
        if (object) { return object->getRGBColor(rgb, color_space, is_linear); }
        return false;
      }
      void setRGBColor(const ColorRGB& rgb, ColorSpace color_space = ColorSpace::eDefault, Bool is_linear = true)
      {
        auto object = getObject();
        if (object) { return object->setRGBColor(rgb, color_space, is_linear); }
      }
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setXYZColor, ColorXYZ);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE(getColorSpace, ColorSpace);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setColorSpace, ColorSpace);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setLinear, Bool);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE(isLinear, Bool);

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
      using impl_type::sample;
      using impl_type::getRGBColor;
      using impl_type::getXYZColor;
      using impl_type::getMinWaveLength;
      using impl_type::getMaxWaveLength;
      using impl_type::getColorSetting;
      using impl_type::setColorSetting;
    };
    struct SpectrumColorSerializer : public ObjectSerializer{
      virtual ~SpectrumColorSerializer() {}
      // ObjectSerializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    struct SpectrumColorDeserializer : public ObjectDeserializer {
      virtual ~SpectrumColorDeserializer() {}
      // ObjectDeserializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const Json& json) const->std::shared_ptr<Object> override;
    };

  }
}
