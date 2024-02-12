#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace spectrum {
    struct SpectrumIrregularObject : public SpectrumObject {
      using base_type = SpectrumObject;
      static Bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "SpectrumIrregular"; };
      static auto create(const Array<Pair<F32, F32>>& values = { {360.f,1.0f},{830.f,1.0f} }) -> std::shared_ptr<SpectrumIrregularObject> {
        return std::shared_ptr<SpectrumIrregularObject>(new SpectrumIrregularObject(values));
      }
      virtual ~SpectrumIrregularObject() noexcept {}
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
      virtual auto getMinWaveLength() const->F32 override;
      virtual auto getMaxWaveLength() const->F32 override;
      auto getWaveLengthsAndIntensities() const->Array<Pair<F32, F32>>;
      void setWaveLengthsAndIntensities(const Array<Pair<F32, F32>>& wavelength_and_intensities);
      auto getWaveLengths() const -> Array<F32>;
      auto getIntensities() const -> Array<F32>;
      auto getIntensity(F32 wavelength) const-> F32;
      void setIntensity(F32 wavelength, F32 intensity);
      void popIntensity(F32 wavelength);
      bool hasIntensity(F32 wavelength) const noexcept;
      auto getSize() const->U32;
    private:
      SpectrumIrregularObject(const Array<Pair<F32, F32>>& values) noexcept
        :SpectrumObject(), m_wave_lengths_and_intensities{ values }
      {
        std::sort(std::begin(m_wave_lengths_and_intensities), std::end(m_wave_lengths_and_intensities),
          [](const auto& l, const auto& r) { return l.first < r.second; });
      }
    private:
      Array<Pair<F32,F32>> m_wave_lengths_and_intensities;
    };
    struct SpectrumIrregular : protected SpectrumImpl<SpectrumIrregularObject> {
      struct IndexAccesorRef {
        IndexAccesorRef(const IndexAccesorRef&) = delete;
        IndexAccesorRef& operator=(const IndexAccesorRef&) = delete;
        IndexAccesorRef(IndexAccesorRef&&) = delete;
        IndexAccesorRef& operator=(IndexAccesorRef&&) = delete;

        operator F32() const {
          auto obj = m_obj.lock();
          if (obj) { return obj->getIntensity(m_wavelength); }
          else { return 0.0f; }
        }
        void operator=(nullptr_t) noexcept {
          auto obj = m_obj.lock();
          if (obj) {
            obj->popIntensity(m_wavelength);
          }
        }
        void operator=(F32 value) noexcept {
          auto obj = m_obj.lock();
          if (obj) {
            obj->setIntensity(m_wavelength, value);
          }
        }
      private:
        IndexAccesorRef(const std::shared_ptr<SpectrumIrregularObject>& obj, F32 wavelength) noexcept
          :m_obj{ obj }, m_wavelength{ wavelength } {}
        friend class SpectrumIrregular;
      private:
        std::weak_ptr<SpectrumIrregularObject> m_obj;
        F32 m_wavelength;
      };
      using impl_type = SpectrumImpl<SpectrumIrregularObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      SpectrumIrregular() noexcept : impl_type() {}
      SpectrumIrregular(const Array<Pair<F32, F32>>& values = { {360.f,1.0f},{830.f,1.0f} }) noexcept : impl_type{ SpectrumIrregularObject::create(values) } {}
      SpectrumIrregular(nullptr_t) noexcept : impl_type(nullptr) {}
      SpectrumIrregular(const std::shared_ptr<SpectrumIrregularObject>& object) noexcept : impl_type(object) {}
      SpectrumIrregular(const SpectrumIrregular& opb) noexcept : impl_type(opb.getObject()) {}
      SpectrumIrregular(SpectrumIrregular&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      SpectrumIrregular& operator=(const SpectrumIrregular& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      SpectrumIrregular& operator=(SpectrumIrregular&& opb) noexcept
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
      SpectrumIrregular& operator=(const std::shared_ptr<SpectrumIrregularObject>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      auto operator[](U32 idx) const -> F32 { return getIntensity(idx); }
      auto operator[](U32 idx) ->IndexAccesorRef { return IndexAccesorRef(getObject(), idx); }

#define HK_ARRAY_PAIR_F32_F32 Array<Pair<F32,F32>>
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getWaveLengthsAndIntensities, HK_ARRAY_PAIR_F32_F32, {});
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setWaveLengthsAndIntensities, HK_ARRAY_PAIR_F32_F32);
#undef  HK_ARRAY_PAIR_F32_F32

      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getIntensities, Array<F32>, {});
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getWaveLengths, Array<F32>, {});
      HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getSize, U32, 0);
      auto getIntensity(F32 wavelength)const->F32 {
        auto object = getObject();
        if (!object) { return 0.0f; }
        return object->getIntensity(wavelength);
      }
      void setIntensity(F32 wavelength, F32 value) {
        auto object = getObject();
        if (!object) { return; }
        return object->setIntensity(wavelength, value);
      }
      void popIntensity(F32 wavelength) {
        auto object = getObject();
        if (!object) { return; }
        return object->popIntensity(wavelength);
      }
      Bool hasIntensity(F32 wavelength) const{
        auto object = getObject();
        if (!object) { return false; }
        return object->hasIntensity(wavelength);
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
    struct SpectrumIrregularSerializer : public ObjectSerializer {
      virtual ~SpectrumIrregularSerializer() noexcept {}
      // ObjectSerializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    struct SpectrumIrregularDeserializer : public ObjectDeserializer {
      virtual ~SpectrumIrregularDeserializer() noexcept {}
      // ObjectDeserializer を介して継承されました
      auto getTypeString() const noexcept -> Str override;
      auto eval(const Json& json) const->std::shared_ptr<Object> override;
    };

  }
}
