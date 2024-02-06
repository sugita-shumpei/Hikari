#include "volume.h"

auto hikari::assets::mitsuba::XMLVolume::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLVolume>
{
  return std::shared_ptr<XMLVolume>(new XMLVolume(context,plugin_type));
}

hikari::assets::mitsuba::XMLVolume::~XMLVolume() noexcept
{
}

hikari::assets::mitsuba::XMLVolume::XMLVolume(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
XMLObject(Type::eVolume,plugin_type),m_context{context}
{
}
