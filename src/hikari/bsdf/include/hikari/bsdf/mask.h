#pragma once

#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfMask : public Bsdf
  {
    static constexpr Uuid ID() { return Uuid::from_string("E945AB37-BD0C-4E6F-B4E4-CC05E2D058FF").value(); }
    static auto create() -> std::shared_ptr<BsdfMask>;
    virtual ~BsdfMask();

    auto getBsdf() -> BsdfPtr;
    void setBsdf(const BsdfPtr& bsdf);

    auto getOpacity() -> SpectrumOrTexture;
    void setOpacity(const SpectrumOrTexture& texture);

    // Bsdf を介して継承されました
    Uuid getID() const override;
  private:
    BsdfMask();
  private:
    BsdfPtr    m_bsdf;
    SpectrumOrTexture m_opacity;
  };
}
