#include "texture.h"

auto hikari::assets::mitsuba::XMLTexture::create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) -> std::shared_ptr<XMLTexture>
{
  auto texture = std::shared_ptr<XMLTexture>(new XMLTexture(context, plugin_type,ref_id));
  context->setRefObject(texture);
  return texture;
}

hikari::assets::mitsuba::XMLTexture::~XMLTexture() noexcept
{
}

hikari::assets::mitsuba::XMLTexture::XMLTexture(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) noexcept
  :XMLReferableObject(Type::eTexture,plugin_type,ref_id),m_context{context}
{
}
