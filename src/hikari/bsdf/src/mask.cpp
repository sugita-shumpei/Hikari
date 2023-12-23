#include <hikari/bsdf/mask.h>

auto hikari::BsdfMask::create(const BsdfPtr& bsdf, const TexturePtr& texture) -> std::shared_ptr<BsdfMask>
{
    return std::shared_ptr<BsdfMask>(new BsdfMask(bsdf,texture));
}

hikari::BsdfMask::~BsdfMask()
{
}

hikari::Uuid hikari::BsdfMask::getID() const
{
    return ID();
}

void hikari::BsdfMask::setOpacity(const SpectrumOrTexture& opacity)
{
  m_opacity = opacity;
}

void hikari::BsdfMask::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdf = bsdf;
}

auto hikari::BsdfMask::getOpacity() const -> SpectrumOrTexture
{
    return m_opacity;
}

auto hikari::BsdfMask::getBsdf() -> BsdfPtr
{
    return m_bsdf;
}

hikari::BsdfMask::BsdfMask(const BsdfPtr& bsdf, const TexturePtr& opacity):Bsdf(),
m_bsdf{bsdf},
m_opacity{ opacity }
{
}
