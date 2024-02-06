#include "rfilter.h"
auto hikari::assets::mitsuba::XMLRFilter::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLRFilter>
{
  return std::shared_ptr<XMLRFilter>(new XMLRFilter(context,plugin_type));
}

hikari::assets::mitsuba::XMLRFilter::~XMLRFilter() noexcept
{
}

hikari::assets::mitsuba::XMLRFilter::XMLRFilter(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
XMLObject(Type::eRFilter,plugin_type),m_context{context}
{
}
