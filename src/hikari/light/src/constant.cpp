#include <hikari/light/constant.h>

auto hikari::LightConstant::create() -> std::shared_ptr<LightConstant>
{
    return std::shared_ptr<LightConstant>(new LightConstant());
}

hikari::LightConstant::~LightConstant()
{
}

hikari::Uuid hikari::LightConstant::getID() const
{
    return ID();
}

void hikari::LightConstant::setRadiance(const SpectrumPtr& radiance)
{
  m_radiance = radiance;
}

auto hikari::LightConstant::getRadiance() const -> SpectrumPtr
{
    return m_radiance;
}

hikari::LightConstant::LightConstant() :Light(), m_radiance{}
{}
