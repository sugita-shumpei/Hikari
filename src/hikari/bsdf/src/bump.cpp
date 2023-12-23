#include <hikari/bsdf/bump.h>

auto hikari::BsdfBump::create(const BsdfPtr& bsdf, const TexturePtr& texture, F32 scale) -> std::shared_ptr<BsdfBump>
{
    return std::shared_ptr<BsdfBump>(new BsdfBump(bsdf,texture,scale));
}

hikari::BsdfBump::~BsdfBump()
{
}

void hikari::BsdfBump::setTexture(const TexturePtr& texture)
{
  m_texture = texture;
}

void hikari::BsdfBump::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdf = bsdf;
}

void hikari::BsdfBump::setScale(F32 scale)
{
  m_scale = scale;
}

auto hikari::BsdfBump::getTexture() -> TexturePtr
{
    return m_texture;
}

auto hikari::BsdfBump::getBsdf() -> BsdfPtr
{
    return m_bsdf;
}

auto hikari::BsdfBump::getScale() const -> F32
{
    return m_scale;
}

hikari::BsdfBump::BsdfBump(const BsdfPtr& bsdf, const TexturePtr& texture, F32 scale)
  :Bsdf(),m_bsdf{bsdf},m_texture{texture},m_scale{scale}
{
}

hikari::Uuid hikari::BsdfBump::getID() const
{
  return ID();
}
