#include <hikari/core/material.h>
#include <hikari/core/node.h>

auto hikari::Material::create() -> std::shared_ptr<Material>
{
    return std::shared_ptr<Material>(new Material());
}

hikari::Material::~Material() noexcept
{
}

auto hikari::Material::getBsdf() -> std::shared_ptr<Bsdf>
{
  return m_bsdf;
}

auto hikari::Material::getInternalMedium() -> std::shared_ptr<Medium>
{
  return m_internal_medium;
}

auto hikari::Material::getExternalMedium() -> std::shared_ptr<Medium>
{
    return m_external_medium;
}

auto hikari::Material::getOpacity() const -> SpectrumOrTexture
{
    return m_opacity;
}

auto hikari::Material::getBumpMap() -> TexturePtr
{
    return m_bump_map;
}

auto hikari::Material::getBumpScale() const -> F32
{
    return m_bump_scale;
}

hikari::Bool hikari::Material::getTwoSided() const
{
  return m_two_sided;
}

void hikari::Material::setBsdf(const std::shared_ptr<Bsdf>& bsdf)
{
  m_bsdf = bsdf;
}

void hikari::Material::setInternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_internal_medium = medium;
}

void hikari::Material::setExternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_external_medium = medium;
}

void hikari::Material::setOpacity(const SpectrumOrTexture& opacity_map)
{
  m_opacity = opacity_map;
}

void hikari::Material::setBumpMap(const TexturePtr& bump_map)
{
  m_bump_map = bump_map;
}

void hikari::Material::setBumpScale(F32 bump_scale)
{
  m_bump_scale = bump_scale;
}

void hikari::Material::setTwoSided(Bool two_sided)
{
  m_two_sided;
}

hikari::Bool hikari::Material::hasBumpMap() const
{
  return m_bump_map!=nullptr;
}

hikari::Bool hikari::Material::HasOpacity() const
{
  return (Bool)m_opacity;
}

hikari::Material::Material() :
  m_bsdf{},
  m_internal_medium{},
  m_external_medium{},
  m_bump_map{nullptr},
  m_bump_scale{1.0f},
  m_opacity{SpectrumPtr(nullptr)},
  m_two_sided{false}
{
}
