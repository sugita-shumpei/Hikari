#include <hikari/bsdf/normal_map.h>

auto hikari::BsdfNormalMap::create() -> std::shared_ptr<BsdfNormalMap>
{
    return std::shared_ptr<BsdfNormalMap>(new BsdfNormalMap());
}

hikari::BsdfNormalMap::~BsdfNormalMap()
{
}

auto hikari::BsdfNormalMap::getBsdf() -> BsdfPtr
{
    return m_bsdf;
}

void hikari::BsdfNormalMap::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdf = bsdf;
}

auto hikari::BsdfNormalMap::getTexture() -> TexturePtr
{
    return m_texture;
}

void hikari::BsdfNormalMap::setTexture(const TexturePtr& texture)
{
  m_texture = texture;
}

hikari::Uuid hikari::BsdfNormalMap::getID() const
{
  return ID();
}

hikari::BsdfNormalMap::BsdfNormalMap()
  :Bsdf(),m_bsdf{nullptr},m_texture{nullptr}
{

}
