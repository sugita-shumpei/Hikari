#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/texture.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfMask : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("899F1B87-B990-4E31-94F3-38DA46E5B9AE").value(); }
    static auto create(const BsdfPtr& bsdf, const TexturePtr& texturem) -> std::shared_ptr<BsdfMask>;
    virtual ~BsdfMask();
    virtual Uuid getID() const override;

    void setOpacity(const SpectrumOrTexture& opacity);
    void setBsdf(const BsdfPtr& bsdf);

    auto getOpacity() const -> SpectrumOrTexture;
    auto getBsdf() -> BsdfPtr;
  private:
    BsdfMask(const BsdfPtr& bsdf, const TexturePtr& opacity);
  private:
    BsdfPtr           m_bsdf;
    SpectrumOrTexture m_opacity;
  };
}
