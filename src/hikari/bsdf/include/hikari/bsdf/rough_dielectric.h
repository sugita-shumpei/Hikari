#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfRoughDielectric : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("95E64E47-1F3C-477C-8D21-88C774B74B83").value(); }
    static auto  create(F32 int_ior = 1.0f, F32 ext_ior = 1.0f) -> std::shared_ptr<BsdfRoughDielectric>;
    virtual ~BsdfRoughDielectric();
    virtual Uuid getID() const override;

    void setIntIOR(F32 int_ior);
    void setExtIOR(F32 ext_ior);

    F32  getIntIOR() const;
    F32  getExtIOR() const;

    auto getSpecularReflectance() const->SpectrumOrTexture;
    auto getSpecularTransmittance() const->SpectrumOrTexture;

    void setSpecularReflectance(const SpectrumOrTexture& ref);
    void setSpecularTransmittance(const SpectrumOrTexture& ref);

    void setDistribution(BsdfDistributionType distribution);
    auto getDistribution() const-> BsdfDistributionType;

    void setAlpha(const FloatOrTexture& alpha);
    auto getAlpha()  const->std::optional<FloatOrTexture>;

    auto getAlphaU() const->FloatOrTexture;
    void setAlphaU(const FloatOrTexture& alpha_u);

    auto getAlphaV() const->FloatOrTexture;
    void setAlphaV(const FloatOrTexture& alpha_v);
  private:
    BsdfRoughDielectric(F32 int_ior, F32 ext_ior);
  private:
    SpectrumOrTexture    m_specular_reflectance  ;
    SpectrumOrTexture    m_specular_transmittance;
    FloatOrTexture       m_alpha_u;
    FloatOrTexture       m_alpha_v;
    BsdfDistributionType m_distribution;
    F32                  m_int_ior;
    F32                  m_ext_ior;
    Bool                 m_anisotropy;
  };
}
