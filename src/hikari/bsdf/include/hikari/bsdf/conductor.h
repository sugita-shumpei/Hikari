#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfConductor : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("54D122F8-80EB-471F-89F9-8FA82870A059").value(); }
    static auto  create() -> std::shared_ptr<BsdfConductor>;
    virtual ~BsdfConductor();
    virtual Uuid getID() const override;

    void setEta(const SpectrumOrTexture& eta);
    void setK  (const SpectrumOrTexture&   k);

    auto getEta() const ->SpectrumOrTexture;
    auto getK()   const ->SpectrumOrTexture;

    auto getSpecularReflectance() const->SpectrumOrTexture;
    void setSpecularReflectance(const SpectrumOrTexture& ref);
  private:
    BsdfConductor();
  private:
    SpectrumOrTexture m_specular_reflectance;
    SpectrumOrTexture m_eta;
    SpectrumOrTexture m_k;
  };
}
