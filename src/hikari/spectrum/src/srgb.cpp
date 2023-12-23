#include <hikari/spectrum/srgb.h>

auto hikari::SpectrumSrgb::create(const Rgb3F& color) -> std::shared_ptr<SpectrumSrgb>
{
  return std::shared_ptr<SpectrumSrgb>(new SpectrumSrgb(color));
}

hikari::SpectrumSrgb::~SpectrumSrgb()
{
}

hikari::Uuid hikari::SpectrumSrgb::getID() const
{
    return ID();
}

void hikari::SpectrumSrgb::setColor(const Rgb3F& color)
{
  m_color = color;
}

auto hikari::SpectrumSrgb::getColor() const -> Rgb3F
{
  return m_color;
}

hikari::SpectrumSrgb::SpectrumSrgb(const Rgb3F& color):Spectrum(),m_color{ color }
{
}
