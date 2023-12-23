#pragma once
#include <hikari/core/variant.h>
#include <hikari/core/bsdf.h>

namespace hikari {
  struct BsdfPhong : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("2623FDAF-88E8-47F0-9601-31AD88FE9181").value(); }
    static auto create() -> std::shared_ptr<BsdfPhong>;
    virtual ~BsdfPhong();

    Uuid getID() const override;

    auto getDiffuseReflectance() const->SpectrumOrTexture;
    void setDiffuseReflectance(const SpectrumOrTexture& ref);

    auto getSpecularReflectance() const->SpectrumOrTexture;
    void setSpecularReflectance(const SpectrumOrTexture& ref);

    auto getExponent() const->FloatOrTexture;
    void setExponent(const FloatOrTexture& exp);
  private:
    BsdfPhong();
  private:
    SpectrumOrTexture m_diffuse_reflectance;
    SpectrumOrTexture m_specular_reflectance;
    FloatOrTexture    m_exponent;
  };
}
