#include "medium.h"

auto hikari::assets::mitsuba::XMLMedium::create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) -> std::shared_ptr<XMLMedium>
{
  auto medium= std::shared_ptr<XMLMedium>(new XMLMedium(context, plugin_type, ref_id));
  context->setRefObject(medium);
  return medium;
}

hikari::assets::mitsuba::XMLMedium::~XMLMedium() noexcept
{
}

auto hikari::assets::mitsuba::XMLMedium::getPhase() const -> std::shared_ptr<XMLPhase>
{
  return std::static_pointer_cast<XMLPhase>(getNestObj(Type::ePhase,0));
}

void hikari::assets::mitsuba::XMLMedium::setPhase(const std::shared_ptr<XMLPhase>& phase)
{
  setNestObj(0, phase);
}

hikari::assets::mitsuba::XMLMedium::XMLMedium(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) noexcept
  :XMLReferableObject(Type::eMedium, plugin_type, ref_id),m_context{context}
{
}
