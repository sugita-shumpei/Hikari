#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace spectrum {
    struct SpectrumUniformObject : public SpectrumObject {
      using base_type = SpectrumObject;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "SpectrumUniform"; };
      static auto create(F32 intensity = 1.0f, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) -> std::shared_ptr<SpectrumUniformObject> {
        return std::shared_ptr<SpectrumUniformObject>(new SpectrumUniformObject(intensity, min_wavelength, max_wavelength));
      }
      virtual ~SpectrumUniformObject() noexcept {}
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
      virtual auto getXYZColor() const->ColorXYZ override;
      virtual auto getMinWaveLength() const->F32 override { return m_min_wavelength; }
      virtual auto getMaxWaveLength() const->F32 override { return m_max_wavelength; }

      void setMinWaveLength(F32 wave_length) { m_min_wavelength = wave_length; }
      void setMaxWaveLength(F32 wave_length) { m_max_wavelength = wave_length; }
      void setIntensity(F32 value) { m_intensity = value; }
      auto getIntensity() const -> F32 { return m_intensity; }
    private:
      SpectrumUniformObject(F32 intensity, F32 min_wavelength, F32 max_wavelength) noexcept
        :SpectrumObject(), m_intensity{ intensity }, m_min_wavelength{ min_wavelength }, m_max_wavelength{ max_wavelength }
      {}
    private:
      F32 m_min_wavelength;
      F32 m_max_wavelength;
      F32 m_intensity;
    };
    struct SpectrumUniform : protected SpectrumImpl<SpectrumUniformObject> {
      using impl_type = SpectrumImpl<SpectrumUniformObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      SpectrumUniform() noexcept : impl_type() {}
      SpectrumUniform(F32 intensity = 1.0f, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) noexcept : impl_type{ SpectrumUniformObject::create(intensity,min_wavelength,max_wavelength) } {}
      SpectrumUniform(nullptr_t) noexcept : impl_type(nullptr) {}
      SpectrumUniform(const std::shared_ptr<SpectrumUniformObject>& object) noexcept : impl_type(object) {}
      SpectrumUniform(const SpectrumUniform& opb) noexcept : impl_type(opb.getObject()) {}
      SpectrumUniform(SpectrumUniform&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      SpectrumUniform& operator=(const SpectrumUniform& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      SpectrumUniform& operator=(SpectrumUniform&& opb) noexcept
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
      SpectrumUniform& operator=(const std::shared_ptr<SpectrumUniformObject>& obj) noexcept
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
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setIntensity, F32);
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE(getIntensity, F32);

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
    struct SpectrumUniformSerializer : public ObjectSerializer {
      virtual ~SpectrumUniformSerializer() noexcept {}
      // ObjectSerializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    struct SpectrumUniformDeserializer : public ObjectDeserializer {
      virtual ~SpectrumUniformDeserializer() noexcept {}
      // ObjectDeserializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const Json& json) const->std::shared_ptr<Object> override;
    };
  }
}
