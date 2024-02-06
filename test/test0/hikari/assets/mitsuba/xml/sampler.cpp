#include "sampler.h"

auto hikari::assets::mitsuba::XMLSampler::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLSampler>
{
  return std::shared_ptr<XMLSampler>(new XMLSampler(context,plugin_type));
}

hikari::assets::mitsuba::XMLSampler::~XMLSampler() noexcept
{
}

hikari::assets::mitsuba::XMLSampler::XMLSampler(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
  XMLObject(Type::eSampler,plugin_type),m_context{context}
{
}
