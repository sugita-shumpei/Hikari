#include <hikari/film/hdr.h>
auto hikari::FilmHdr::create(U32 width , U32 height ) -> std::shared_ptr<FilmHdr> {
  return std::shared_ptr<FilmHdr>(new FilmHdr(width, height));
}
hikari::FilmHdr::~FilmHdr(){}

auto hikari::FilmHdr::getWidth() const->U32 { return m_width; }
auto hikari::FilmHdr::getHeight() const->U32 { return m_height; }

void hikari::FilmHdr::setWidth(U32 width) {   m_width = width; }
void hikari::FilmHdr::setHeight(U32 height) { m_height = height; }

void hikari::FilmHdr::setComponentFormat(FilmComponentFormat component_format) {
  m_component_format = component_format;
}
auto hikari::FilmHdr::getComponentFormat() const->FilmComponentFormat {
  return m_component_format;
}

void hikari::FilmHdr::setPixelFormat(FilmPixelFormat pixel_format) {
  m_pixel_format = pixel_format;
}
auto hikari::FilmHdr::getPixelFormat() const->FilmPixelFormat {
  return m_pixel_format;
}

hikari::FilmHdr::FilmHdr(U32 width, U32 height)
  :Film(),m_width{ width }, m_height{ height }, m_pixel_format{ FilmPixelFormat::eRGB }, m_component_format{ FilmComponentFormat::eFloat16 } {}

hikari::Uuid hikari::FilmHdr::getID() const
{
  return ID();
}
