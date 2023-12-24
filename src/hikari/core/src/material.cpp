#include <hikari/core/material.h>
#include <hikari/core/node.h>

auto hikari::Material::create() -> std::shared_ptr<Material>
{
    return std::shared_ptr<Material>(new Material());
}

hikari::Material::~Material() noexcept
{
}

auto hikari::Material::getBsdf() -> std::shared_ptr<Bsdf> { return m_bsdf; }


auto hikari::Material::getInternalMedium() -> std::shared_ptr<Medium>
{
  return m_internal_medium;
}

auto hikari::Material::getExternalMedium() -> std::shared_ptr<Medium>
{
    return m_external_medium;
}


void hikari::Material::setBsdf(const std::shared_ptr<Bsdf>& bsdf) { m_bsdf = bsdf; }

void hikari::Material::setInternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_internal_medium = medium;
}

void hikari::Material::setExternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_external_medium = medium;
}


hikari::Material::Material() :
  m_bsdf{nullptr},
  m_internal_medium{},
  m_external_medium{}
{
}
