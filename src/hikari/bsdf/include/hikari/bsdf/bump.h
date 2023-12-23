#pragma once
#include <hikari/core/bsdf.h>
#include <hikari/core/texture.h>
namespace hikari {
  struct BsdfBump : public Bsdf{
    static constexpr Uuid ID() { return Uuid::from_string("C82A615E-E568-4AE6-B5F2-8E2534D06454").value(); }
    static auto create(const BsdfPtr& bsdf,const TexturePtr& texturem, F32 scale = 1.0f) -> std::shared_ptr<BsdfBump>;
    virtual ~BsdfBump();
    virtual Uuid getID() const override;

    void setTexture(const TexturePtr& texture);
    void setBsdf(const BsdfPtr& bsdf);
    void setScale(F32  scale);

    auto getTexture() -> TexturePtr;
    auto getBsdf() -> BsdfPtr;
    auto getScale()const->F32;
  private:
    BsdfBump(const BsdfPtr& bsdf, const TexturePtr& texture, F32 scale);
  private:
    TexturePtr m_texture;
    BsdfPtr    m_bsdf ;
    F32        m_scale;
  };
}
