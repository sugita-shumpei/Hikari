#pragma once
#include <hikari/core/variant.h>
#include <hikari/core/bsdf.h>

namespace hikari {
  struct BsdfPlastic : public Bsdf{
    static constexpr Uuid ID() { return Uuid::from_string("9E01FE9D-FDE1-41F6-872D-07364181B0E8").value(); }
    static auto create() -> std::shared_ptr<BsdfPlastic>;
    virtual ~BsdfPlastic();

    Uuid getID() const override;

    void setIntIOR(F32 int_ior);
    void setExtIOR(F32 ext_ior);

    F32  getIntIOR() const;
    F32  getExtIOR() const;

    F32  getEta() const;
    F32  getIntFresnelDiffuseReflectance() const;
    F32  getExtFresnelDiffuseReflectance() const;

    auto getDiffuseReflectance() const->SpectrumOrTexture;
    void setDiffuseReflectance(const SpectrumOrTexture& ref);

    auto getSpecularReflectance() const->SpectrumOrTexture;
    void setSpecularReflectance(const SpectrumOrTexture& ref);

    Bool getNonLinear() const;
    void setNonLinear(Bool non_linear);
  private:
    BsdfPlastic();
    static auto CalculateFresnelDiffuseReflectance(F32 eta) -> F32;
    static auto calculateReflection(F32 sin_in, F32 cos_in, F32 eta)  -> F32;
  private:
    SpectrumOrTexture m_diffuse_reflectance;
    SpectrumOrTexture m_specular_reflectance;
    F32               m_int_ior;
    F32               m_ext_ior;
    F32               m_int_fresnel_diffuse_reflectance;
    F32               m_ext_fresnel_diffuse_reflectance;
    Bool              m_nonlinear;
  };
}
