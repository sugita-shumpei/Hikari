#include "film.h"
;

auto hikari::assets::mitsuba::XMLFilm::create(const XMLContextPtr& context, const std::string& plugin_type) -> std::shared_ptr<XMLFilm>
{
  return std::shared_ptr<XMLFilm>(new XMLFilm(context,plugin_type));
}

hikari::assets::mitsuba::XMLFilm::~XMLFilm() noexcept
{
}

hikari::assets::mitsuba::XMLFilm::XMLFilm(const XMLContextPtr& context, const std::string& plugin_type) noexcept:
XMLObject(Type::eFilm,plugin_type),m_context{context}
{
}
