#include "spectrum.h"

auto hikari::assets::mitsuba::XMLSpectrum::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSpectrum>
{
  return std::shared_ptr<XMLSpectrum>(new XMLSpectrum(context, plugin_type));
}

hikari::assets::mitsuba::XMLSpectrum::~XMLSpectrum() noexcept
{
}

hikari::assets::mitsuba::XMLSpectrum::XMLSpectrum(const XMLContextPtr& context, const std::string& plugin_type) noexcept
  :XMLObject(Type::eSpectrum,plugin_type),m_context{context}
{
}
