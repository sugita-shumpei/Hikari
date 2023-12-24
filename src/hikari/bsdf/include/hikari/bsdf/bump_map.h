#pragma once

#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfBumpMap : public Bsdf
  {
    static constexpr Uuid ID() { return Uuid::from_string("DF620FA9-F126-4982-90CD-01A405AB0324").value(); }
    static auto create() -> std::shared_ptr<BsdfBumpMap>;
    virtual ~BsdfBumpMap();

    auto getBsdf() -> BsdfPtr;
    void setBsdf(const BsdfPtr& bsdf);

    auto getTexture() -> TexturePtr;
    void setTexture(const TexturePtr& texture);

    auto getScale() const->F32;
    void setScale(F32 scale);

    // Bsdf を介して継承されました
    Uuid getID() const override;
  private:
    BsdfBumpMap();
  private:
    BsdfPtr    m_bsdf;
    TexturePtr m_texture;
    F32        m_scale;
  };
}
