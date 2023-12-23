#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfDiffuse : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("61014F7E-5F0A-48D3-A036-75EF71359257").value(); }
    static auto  create() -> std::shared_ptr<BsdfDiffuse>;
    static auto  create(const SpectrumOrTexture& ref) -> std::shared_ptr<BsdfDiffuse>;
    virtual ~BsdfDiffuse();
    virtual Uuid getID() const override;
    auto getReflectance() const->SpectrumOrTexture   ;
    void setReflectance(const SpectrumOrTexture& ref);
  private:
    BsdfDiffuse(const SpectrumOrTexture& ref);
  private:
    SpectrumOrTexture m_reflectance;
  };
}
