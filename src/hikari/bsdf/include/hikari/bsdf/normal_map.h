#pragma once

#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfNormalMap : public Bsdf
  {
    static constexpr Uuid ID() { return Uuid::from_string("E8BC43C0-16B6-4AF5-989B-601427B52302").value(); }
    static auto create() -> std::shared_ptr<BsdfNormalMap>;
    virtual ~BsdfNormalMap();

    auto getBsdf() -> BsdfPtr;
    void setBsdf(const BsdfPtr& bsdf);

    auto getTexture() -> TexturePtr;
    void setTexture(const TexturePtr& texture);

    // Bsdf を介して継承されました
    Uuid getID() const override;
  private:
    BsdfNormalMap();
  private:
    BsdfPtr    m_bsdf;
    TexturePtr m_texture;
  };
}
