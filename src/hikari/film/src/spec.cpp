#include <hikari/film/spec.h>
auto hikari::FilmSpec::create(U32 width, U32 height) -> std::shared_ptr<FilmSpec> {
  return std::shared_ptr<FilmSpec>(new FilmSpec(width, height));
}
hikari::FilmSpec::~FilmSpec() {}

auto hikari::FilmSpec::getWidth() const->U32 { return m_width; }
auto hikari::FilmSpec::getHeight() const->U32 { return m_height; }

void hikari::FilmSpec::setWidth(U32 width) { m_width = width; }
void hikari::FilmSpec::setHeight(U32 height) { m_height = height; }

void hikari::FilmSpec::setComponentFormat(FilmComponentFormat component_format) {
  m_component_format = component_format;
}
auto hikari::FilmSpec::getComponentFormat() const->FilmComponentFormat {
  return m_component_format;
}


hikari::FilmSpec::FilmSpec(U32 width, U32 height)
  :Film(), m_width{ width }, m_height{ height }, m_component_format{ FilmComponentFormat::eFloat16 } {}

hikari::Uuid hikari::FilmSpec::getID() const
{
  return ID();
}
