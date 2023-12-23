#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfRoughConductor : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("B7740D96-0DF9-458C-BC6B-3A7F998CF597").value(); }
    static auto  create() -> std::shared_ptr<BsdfRoughConductor>;
    virtual ~BsdfRoughConductor();
    virtual Uuid getID() const override;

    void setEta(const SpectrumOrTexture& eta);
    void setK(const SpectrumOrTexture& k);

    auto getEta() const->SpectrumOrTexture;
    auto getK()   const->SpectrumOrTexture;

    auto getSpecularReflectance() const->SpectrumOrTexture;
    void setSpecularReflectance(const SpectrumOrTexture& ref);

    void setDistribution(BsdfDistributionType distribution);
    auto getDistribution() const->BsdfDistributionType;

    void setAlpha(const FloatOrTexture& alpha);
    auto getAlpha()  const->std::optional<FloatOrTexture>;

    auto getAlphaU() const->FloatOrTexture;
    void setAlphaU(const FloatOrTexture& alpha_u);

    auto getAlphaV() const->FloatOrTexture;
    void setAlphaV(const FloatOrTexture& alpha_v);
  private:
    BsdfRoughConductor();
  private:
    SpectrumOrTexture    m_specular_reflectance;
    SpectrumOrTexture    m_eta;
    SpectrumOrTexture    m_k;
    FloatOrTexture       m_alpha_u;
    FloatOrTexture       m_alpha_v;
    BsdfDistributionType m_distribution;
    Bool                 m_anisotropy;
  };
}
