#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfThinDielectric : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("4290E1D7-1C37-4B70-84A8-9FEEFBF75737").value(); }
    static auto  create(F32 int_ior = 1.0f, F32 ext_ior = 1.0f) -> std::shared_ptr<BsdfThinDielectric>;
    virtual ~BsdfThinDielectric();
    virtual Uuid getID() const override { return ID(); }

    void setIntIOR(F32 int_ior);
    void setExtIOR(F32 ext_ior);

    F32  getIntIOR() const;
    F32  getExtIOR() const;

    F32 getEta() const;

    auto getSpecularReflectance() const->SpectrumOrTexture;
    auto getSpecularTransmittance() const->SpectrumOrTexture;

    void setSpecularReflectance(const SpectrumOrTexture& ref);
    void setSpecularTransmittance(const SpectrumOrTexture& ref);
  private:
    BsdfThinDielectric(F32 int_ior, F32 ext_ior);
  private:
    F32               m_int_ior;
    F32               m_ext_ior;
    SpectrumOrTexture m_specular_reflectance;
    SpectrumOrTexture m_specular_transmittance;
  };
}
