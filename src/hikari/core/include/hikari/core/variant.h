#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/texture.h>
#include <variant>
namespace hikari {
  struct SpectrumOrTexture {
    SpectrumOrTexture() :m_value{} {}
    SpectrumOrTexture(const SpectrumOrTexture&)            = default;
    SpectrumOrTexture& operator=(const SpectrumOrTexture&) = default;
    SpectrumOrTexture(const SpectrumPtr& spec):m_value{ spec } {
      if (!spec) {
        m_value = {};
      }
    }
    SpectrumOrTexture(const TexturePtr& text) :m_value{ text } {
      if (!text) {
        m_value = {};
      }
    }
    template<typename DeriveType, std::enable_if_t<std::is_base_of_v<Spectrum, DeriveType>, nullptr_t> = nullptr>
    SpectrumOrTexture(const std::shared_ptr<DeriveType>& spec) : SpectrumOrTexture(std::static_pointer_cast<Spectrum>(spec)) {}
    template<typename DeriveType, std::enable_if_t<std::is_base_of_v<Texture , DeriveType>, nullptr_t> = nullptr>
    SpectrumOrTexture(const  std::shared_ptr<DeriveType>& text ) : SpectrumOrTexture(std::static_pointer_cast<Texture>(text)) {}

    explicit operator Bool() const {  return m_value.index()==2; }
    Bool operator!() const { return m_value.index() != 0; }

    void setSpectrum(const SpectrumPtr& spec) { if (!spec) { m_value = {}; return; } m_value = spec; }
    void setTexture (const TexturePtr & text ){ if (!text) { m_value = {}; return; } m_value = text; }

    auto getSpectrum()const->SpectrumPtr {
      if (m_value.index() == 0) { return std::get<0>(m_value); }
      else { return nullptr; }
    }
    auto getTexture ()const-> TexturePtr{
      if (m_value.index() == 1) { return std::get<1>(m_value); }
      else { return nullptr; }
    }
  private:
    std::variant<SpectrumPtr, TexturePtr, std::monostate> m_value;
  };
  struct FloatOrTexture {
    FloatOrTexture() :m_value{ nullptr } {}
    FloatOrTexture(const FloatOrTexture&) = default;
    FloatOrTexture& operator=(const FloatOrTexture&) = default;
    FloatOrTexture(F32              value) :m_value{ value } {}
    FloatOrTexture(const TexturePtr& text) :m_value{ text } {}
    template<typename DeriveType, std::enable_if_t<std::is_base_of_v<Texture, DeriveType>, nullptr_t> = nullptr>
    FloatOrTexture(const  std::shared_ptr<DeriveType>& text) : SpectrumOrTexture(std::static_pointer_cast<Texture>(text)) {}

    void setFloat  (F32             value) { m_value = value; }
    void setTexture(const TexturePtr& text) { m_value = text; }

    auto getFloat  ()const->F32 {
      if (m_value.index() == 0) { return std::get<0>(m_value); }
      else { return 0.0f; }
    }
    auto getTexture()const-> TexturePtr {
      if (m_value.index() == 1) { return std::get<1>(m_value); }
      else { return nullptr; }
    }
  private:
    std::variant<F32, TexturePtr> m_value;
  };
}
