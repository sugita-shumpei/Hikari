#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfDielectric : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("8AD09501-529D-440C-8BF4-B8A42814A174").value(); }
    static auto  create(F32 int_ior = 1.0f, F32 ext_ior = 1.0f) -> std::shared_ptr<BsdfDielectric>;
    virtual ~BsdfDielectric();
    virtual Uuid getID() const override { return ID(); }

    void setIntIOR(F32 int_ior);
    void setExtIOR(F32 ext_ior);

    F32  getIntIOR() const;
    F32  getExtIOR() const;
    F32  getEta() const;

    auto getSpecularReflectance() const->SpectrumOrTexture;
    auto getSpecularTransmittance() const->SpectrumOrTexture;

    void setSpecularReflectance(const SpectrumOrTexture& ref);
    void setSpecularTransmittance(const SpectrumOrTexture& ref);
  private:
    BsdfDielectric(F32 int_ior, F32 ext_ior);
  private:
    SpectrumOrTexture m_specular_reflectance;
    SpectrumOrTexture m_specular_transmittance;
    F32               m_int_ior;
    F32               m_ext_ior;
  };
}
