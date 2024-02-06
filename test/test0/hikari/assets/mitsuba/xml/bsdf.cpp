#include "bsdf.h"

auto hikari::assets::mitsuba::XMLBsdf::create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id) -> std::shared_ptr<XMLBsdf>
{
  auto bsdf = std::shared_ptr<XMLBsdf>(new XMLBsdf(context,plugin_type,id));
  context->setRefObject(bsdf);
  return bsdf;
}

auto hikari::assets::mitsuba::XMLBsdf::getContext() const -> XMLContextPtr
{
  return m_context.lock();
}

hikari::assets::mitsuba::XMLBsdf::XMLBsdf(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id) noexcept : XMLReferableObject(Type::eBsdf, plugin_type, id), m_context{context} {}

hikari::assets::mitsuba::XMLBsdf::~XMLBsdf() noexcept {}
