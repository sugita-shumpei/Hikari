#include <hikari/core/material.h>
#include <hikari/core/node.h>

auto hikari::Material::create() -> std::shared_ptr<Material>
{
    return std::shared_ptr<Material>(new Material());
}

hikari::Material::~Material() noexcept
{
}

auto hikari::Material::getSurface() -> std::shared_ptr<Surface> { return m_surface; }


auto hikari::Material::getInternalMedium() -> std::shared_ptr<Medium>
{
  return m_internal_medium;
}

auto hikari::Material::getExternalMedium() -> std::shared_ptr<Medium>
{
    return m_external_medium;
}


void hikari::Material::setSurface(const std::shared_ptr<Surface>& surface) { m_surface = surface; }

void hikari::Material::setInternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_internal_medium = medium;
}

void hikari::Material::setExternalMedium(const std::shared_ptr<Medium>& medium)
{
  m_external_medium = medium;
}


hikari::Material::Material() :
  m_surface{nullptr},
  m_internal_medium{},
  m_external_medium{}
{
}
