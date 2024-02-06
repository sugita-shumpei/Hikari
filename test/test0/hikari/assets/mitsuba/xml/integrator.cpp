#include "integrator.h"

auto hikari::assets::mitsuba::XMLIntegrator::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLIntegrator>
{
  return std::shared_ptr<XMLIntegrator>(new XMLIntegrator(context,plugin_type));
}

hikari::assets::mitsuba::XMLIntegrator::~XMLIntegrator() noexcept
{
}

hikari::assets::mitsuba::XMLIntegrator::XMLIntegrator(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
  XMLObject(Type::eIntegrator,plugin_type),m_context{context}
{
}
