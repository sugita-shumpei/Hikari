#include <hikari/light/area.h>
#include <hikari/core/node.h>

auto hikari::LightArea::create() -> std::shared_ptr<LightArea>
{
  return std::shared_ptr<LightArea>(new LightArea());
}

hikari::LightArea::~LightArea()
{
}

auto hikari::LightArea::getShape() -> std::shared_ptr<Shape>
{
  auto node = getNode();
  if (node) {
    return node->getShape();
  }
  return nullptr;
}

hikari::LightArea::LightArea() : Light(), m_radiance{}
{
}

hikari::Uuid hikari::LightArea::getID() const
{
  return ID();
}

void hikari::LightArea::setRadiance(const SpectrumOrTexture& radiance)
{
  m_radiance = radiance;
}

auto hikari::LightArea::getRadiance() const -> SpectrumOrTexture
{
    return m_radiance;
}
