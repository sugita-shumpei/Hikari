#include "emitter.h"
auto hikari::assets::mitsuba::XMLEmitter::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLEmitter>
{
  return std::shared_ptr<XMLEmitter>(new XMLEmitter(context,plugin_type));
}

hikari::assets::mitsuba::XMLEmitter::~XMLEmitter() noexcept
{
}

hikari::assets::mitsuba::XMLEmitter::XMLEmitter(const XMLContextPtr& context, const std::string& plugin_type) noexcept
  :XMLObject(Type::eEmitter,plugin_type),m_context{context}
{
}
