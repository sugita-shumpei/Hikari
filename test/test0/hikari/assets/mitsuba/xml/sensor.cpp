#include "sensor.h"

auto hikari::assets::mitsuba::XMLSensor::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSensor>
{
  return std::shared_ptr<XMLSensor>(new XMLSensor(context,plugin_type));
}

hikari::assets::mitsuba::XMLSensor::~XMLSensor() noexcept
{
}

void hikari::assets::mitsuba::XMLSensor::setMedium(const std::shared_ptr<XMLMedium>& medium)
{
  setNestObj(0, medium);
}

auto hikari::assets::mitsuba::XMLSensor::getMedium() const -> std::shared_ptr<XMLMedium>
{
  return std::static_pointer_cast<XMLMedium>(getNestObj(Type::eMedium,0));
}

void hikari::assets::mitsuba::XMLSensor::setSampler(const std::shared_ptr<XMLSampler>& sampler)
{
  setNestObj(0, sampler);
  
}

auto hikari::assets::mitsuba::XMLSensor::getSampler() const -> std::shared_ptr<XMLSampler>
{
  return std::static_pointer_cast<XMLSampler>(getNestObj(Type::eSampler, 0));
}

void hikari::assets::mitsuba::XMLSensor::setFilm(const std::shared_ptr<XMLFilm>& film)
{
  setNestObj(0, film);
}

auto hikari::assets::mitsuba::XMLSensor::getFilm() const -> std::shared_ptr<XMLFilm>
{
  return std::static_pointer_cast<XMLFilm>(getNestObj(Type::eFilm,0));
}

hikari::assets::mitsuba::XMLSensor::XMLSensor(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
XMLObject(Type::eSensor, plugin_type),m_context{context}
{
}
