#include <hikari/light/directional.h>

auto hikari::LightDirectional::create() -> std::shared_ptr<LightDirectional>
{
  return std::shared_ptr<LightDirectional>(new LightDirectional());
}

hikari::LightDirectional::~LightDirectional()
{
}

hikari::Uuid hikari::LightDirectional::getID() const
{
  return ID();
}

void hikari::LightDirectional::setIrradiance(const SpectrumPtr& radiance)
{
  m_irradiance = radiance;
}

auto hikari::LightDirectional::getIrradiance() const -> SpectrumPtr
{
  return m_irradiance;
}

void hikari::LightDirectional::setDirection(const Vec3& direction)
{
  m_direction = direction;
}

auto hikari::LightDirectional::getDirection() const -> std::optional<Vec3>
{
  return m_direction;
}

hikari::LightDirectional::LightDirectional() :Light(), m_direction{}, m_irradiance{}
{
}

