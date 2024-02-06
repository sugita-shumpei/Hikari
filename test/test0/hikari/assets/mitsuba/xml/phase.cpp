#include "phase.h"
auto hikari::assets::mitsuba::XMLPhase::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLPhase>
{
  return std::shared_ptr<XMLPhase>(new XMLPhase(context,plugin_type));
}

hikari::assets::mitsuba::XMLPhase::~XMLPhase() noexcept
{
}

hikari::assets::mitsuba::XMLPhase::XMLPhase(const XMLContextPtr& context, const std::string& plugin_type) noexcept
  :XMLObject(Type::ePhase,plugin_type),m_context{context}
{
}
