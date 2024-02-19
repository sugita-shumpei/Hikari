#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace spectrum {
    struct SpectrumRegularObject : public SpectrumObject {
      using base_type = SpectrumObject;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "SpectrumRegular"; };
      static auto create(const Array<F32>& intensities = { 1.0f }, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) -> std::shared_ptr<SpectrumRegularObject> {
        return std::shared_ptr<SpectrumRegularObject>(new SpectrumRegularObject(intensities, min_wavelength, max_wavelength));
      }
      virtual ~SpectrumRegularObject() noexcept {}
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
      auto getIntensities() const noexcept -> Array<F32>;
      void setIntensities(const Array<F32>& intensities);
      void setIntensityCount(U32 size);
      auto getIntensityCount() const->U32;
      auto getIntensity(U32 index)const->F32;
      void setIntensity(U32 index, F32 value);
      void addIntensity(F32 value);
      void popIntensity(U32 index);
    private:
      SpectrumRegularObject(const Array<F32> intensities, F32 min_wavelength, F32 max_wavelength) noexcept
        :SpectrumObject(), m_intensities{ intensities }, m_min_wavelength{ min_wavelength }, m_max_wavelength{ max_wavelength }
      {}
    private:
      F32 m_min_wavelength;
      F32 m_max_wavelength;
      Array<F32> m_intensities;
    };
    struct SpectrumRegular : protected SpectrumImpl<SpectrumRegularObject> {
      struct IndexAccesorRef {
        IndexAccesorRef(const IndexAccesorRef&) = delete;
        IndexAccesorRef& operator=(const IndexAccesorRef&) = delete;
        IndexAccesorRef(IndexAccesorRef&&) = delete;
        IndexAccesorRef& operator=(IndexAccesorRef&&) = delete;

        operator F32() const {
          auto obj = m_obj.lock();
          if (obj) { return obj->getIntensity(m_idx); }
          else { return 0.0f; }
        }
        void operator=(F32 value) noexcept {
          auto obj = m_obj.lock();
          if (obj) {
            obj->setIntensity(m_idx, value);
          }
        }
      private:
        IndexAccesorRef(const std::shared_ptr<SpectrumRegularObject>& obj, U32 idx) noexcept
          :m_obj{ obj }, m_idx{ idx } {}
        friend class SpectrumRegular;
      private:
        std::weak_ptr<SpectrumRegularObject> m_obj;
        U32 m_idx;
      };
      using impl_type = SpectrumImpl<SpectrumRegularObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      SpectrumRegular() noexcept : impl_type() {}
      SpectrumRegular(const Array<F32>& intensities = { 1.0f,1.0f }, F32 min_wavelength = 360.0f, F32 max_wavelength = 830.0f) noexcept : impl_type{ SpectrumRegularObject::create(intensities,min_wavelength,max_wavelength) } {}
      SpectrumRegular(nullptr_t) noexcept : impl_type(nullptr) {}
      SpectrumRegular(const std::shared_ptr<SpectrumRegularObject>& object) noexcept : impl_type(object) {}
      SpectrumRegular(const SpectrumRegular& opb) noexcept : impl_type(opb.getObject()) {}
      SpectrumRegular(SpectrumRegular&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      SpectrumRegular& operator=(const SpectrumRegular& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      SpectrumRegular& operator=(SpectrumRegular&& opb) noexcept
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
      SpectrumRegular& operator=(const std::shared_ptr<SpectrumRegularObject>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      HK_METHOD_OVERLOAD_COMPARE_OPERATORS(SpectrumRegular);

      auto operator[](U32 idx) const -> F32 { return getIntensity(idx); }
      auto operator[](U32 idx) ->IndexAccesorRef { return IndexAccesorRef(getObject(), idx); }

      void setSize(U32 size)
      {
        setIntensityCount(size);
      }
      auto getSize() const -> U32
      {
        return getIntensityCount();
      }
      HK_METHOD_OVERLOAD_SETTER_LIKE(setMinWaveLength, F32);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setMaxWaveLength, F32);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setIntensities, Array<F32>);
      HK_METHOD_OVERLOAD_GETTER_LIKE(getIntensities, Array<F32>);
      HK_METHOD_OVERLOAD_SETTER_LIKE(setIntensityCount, U32);
      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getIntensityCount, U32, 0);
      auto getIntensity(U32 index)const->F32 {
        auto object = getObject();
        if (!object) { return 0.0f; }
        return object->getIntensity(index);
      }
      void setIntensity(U32 index, F32 value) {
        auto object = getObject();
        if (!object) { return; }
        return object->setIntensity(index, value);
      }
      void addIntensity(F32 value) {
        auto object = getObject();
        if (!object) { return; }
        return object->addIntensity(value);
      }
      void popIntensity(U32 index) {
        auto object = getObject();
        if (!object) { return; }
        return object->popIntensity(index);
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
      using impl_type::sample;
      using impl_type::getRGBColor;
      using impl_type::getXYZColor;
      using impl_type::getMinWaveLength;
      using impl_type::getMaxWaveLength;
      using impl_type::getColorSetting;
      using impl_type::setColorSetting;
    };
    struct SpectrumRegularSerializer : public ObjectSerializer {
      virtual ~SpectrumRegularSerializer() noexcept {}
      // ObjectSerializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    struct SpectrumRegularDeserializer : public ObjectDeserializer{
      virtual ~SpectrumRegularDeserializer() noexcept {}
      // ObjectDeserializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const Json& json) const->std::shared_ptr<Object> override;
    };
  }
}
