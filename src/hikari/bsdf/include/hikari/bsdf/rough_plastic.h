#pragma once
#include <hikari/core/variant.h>
#include <hikari/core/bsdf.h>
namespace hikari {
  struct BsdfRoughPlastic : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("F5C0F200-9B17-44F4-826D-551B6124E68E").value(); }
    static auto create() -> std::shared_ptr<BsdfRoughPlastic>;
    virtual ~BsdfRoughPlastic();

    Uuid getID() const override;

    void setIntIOR(F32 int_ior);
    void setExtIOR(F32 ext_ior);

    F32  getIntIOR() const;
    F32  getExtIOR() const;

    auto getDiffuseReflectance() const->SpectrumOrTexture;
    void setDiffuseReflectance(const SpectrumOrTexture& ref);

    auto getSpecularReflectance() const->SpectrumOrTexture;
    void setSpecularReflectance(const SpectrumOrTexture& ref);

    Bool getNonLinear() const;
    void setNonLinear(Bool non_linear);

    void setDistribution(BsdfDistributionType distribution);
    auto getDistribution() const->BsdfDistributionType;

    void setAlpha(const FloatOrTexture& alpha);
    auto getAlpha()  const->std::optional<FloatOrTexture>;

    auto getAlphaU() const->FloatOrTexture;
    void setAlphaU(const FloatOrTexture& alpha_u);

    auto getAlphaV() const->FloatOrTexture;
    void setAlphaV(const FloatOrTexture& alpha_v);
  private:
    BsdfRoughPlastic();
  private:
    SpectrumOrTexture    m_diffuse_reflectance;
    SpectrumOrTexture    m_specular_reflectance;
    FloatOrTexture       m_alpha_u;
    FloatOrTexture       m_alpha_v;
    BsdfDistributionType m_distribution;
    F32                  m_int_ior;
    F32                  m_ext_ior;
    Bool                 m_nonlinear;
    Bool                 m_anisotropy;
  };
}
