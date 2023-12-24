#include <hikari/bsdf/bump_map.h>

hikari::Uuid hikari::BsdfBumpMap::getID() const
{
    return ID();
}

hikari::BsdfBumpMap::BsdfBumpMap()
  : Bsdf(),m_bsdf{nullptr},m_texture{nullptr},m_scale{1.0f}
{
}

auto hikari::BsdfBumpMap::create() -> std::shared_ptr<BsdfBumpMap>
{
  return std::shared_ptr<BsdfBumpMap>(new BsdfBumpMap());
}

hikari::BsdfBumpMap::~BsdfBumpMap()
{
}

auto hikari::BsdfBumpMap::getBsdf() -> BsdfPtr
{
  return m_bsdf;
}

void hikari::BsdfBumpMap::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdf = bsdf;
}

auto hikari::BsdfBumpMap::getTexture() -> TexturePtr
{
  return m_texture;
}

void hikari::BsdfBumpMap::setTexture(const TexturePtr& texture)
{
  m_texture = texture;
}

auto hikari::BsdfBumpMap::getScale() const -> F32
{
  return m_scale;
}

void hikari::BsdfBumpMap::setScale(F32 scale)
{
  m_scale = scale;
}
