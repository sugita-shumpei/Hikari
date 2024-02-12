#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace spectrum {
    struct SpectrumBlackbodyObject : public SpectrumObject {
      using base_type = SpectrumObject;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "SpectrumBlackbody"; };
      static auto create(F32 temperature = 1.0f, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) -> std::shared_ptr<SpectrumBlackbodyObject> {
        return std::shared_ptr<SpectrumBlackbodyObject>(new SpectrumBlackbodyObject(temperature, min_wavelength, max_wavelength));
      }
      virtual ~SpectrumBlackbodyObject() noexcept {}
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
      virtual auto getMinWaveLength() const->F32 override { return m_min_wavelength; }
      virtual auto getMaxWaveLength() const->F32 override { return m_max_wavelength; }

      void setMinWaveLength(F32 wave_length) { m_min_wavelength = wave_length; }
      void setMaxWaveLength(F32 wave_length) { m_max_wavelength = wave_length; }
      void setTemperature(F32 value) { m_temperature = value; }
      auto getTemperature() const -> F32 { return m_temperature; }
    private:
      SpectrumBlackbodyObject(F32 temperature, F32 min_wavelength, F32 max_wavelength) noexcept
        :SpectrumObject(), m_temperature{ temperature }, m_min_wavelength{ min_wavelength }, m_max_wavelength{ max_wavelength }
      {}
    private:
      F32 m_min_wavelength;
      F32 m_max_wavelength;
      F32 m_temperature;
    };
    struct SpectrumBlackbody : protected SpectrumImpl<SpectrumBlackbodyObject> {
      using impl_type = SpectrumImpl<SpectrumBlackbodyObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      SpectrumBlackbody() noexcept : impl_type() {}
      SpectrumBlackbody(F32 intensity = 1.0f, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) noexcept : impl_type{ SpectrumBlackbodyObject::create(intensity,min_wavelength,max_wavelength) } {}
      SpectrumBlackbody(nullptr_t) noexcept : impl_type(nullptr) {}
      SpectrumBlackbody(const std::shared_ptr<SpectrumBlackbodyObject>& object) noexcept : impl_type(object) {}
      SpectrumBlackbody(const SpectrumBlackbody& opb) noexcept : impl_type(opb.getObject()) {}
      SpectrumBlackbody(SpectrumBlackbody&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      SpectrumBlackbody& operator=(const SpectrumBlackbody& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      SpectrumBlackbody& operator=(SpectrumBlackbody&& opb) noexcept
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
      SpectrumBlackbody& operator=(const std::shared_ptr<SpectrumBlackbodyObject>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setMinWaveLength, F32);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setMaxWaveLength, F32);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setTemperature, F32);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE(getTemperature, F32);

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
    struct SpectrumBlackbodySerializer {};
    struct SpectrumBlackbodyDeserializer {};

  }
}
