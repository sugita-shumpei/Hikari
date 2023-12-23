#include <hikari/light/envmap.h>

auto hikari::LightEnvmap::create() -> std::shared_ptr<LightEnvmap>
{
  return std::shared_ptr<LightEnvmap>(new LightEnvmap());
}

hikari::LightEnvmap::~LightEnvmap()
{
}

hikari::Uuid hikari::LightEnvmap::getID() const
{
  return ID();
}

void hikari::LightEnvmap::setBitmap(const BitmapPtr& bitmap)
{
  m_bitmap = bitmap;
}

auto hikari::LightEnvmap::getBitmap() const -> BitmapPtr
{
    return m_bitmap;
}

void hikari::LightEnvmap::setScale(F32 scale)
{
  m_scale = scale;
}

auto hikari::LightEnvmap::getScale() const -> F32
{
    return m_scale;
}

hikari::LightEnvmap::LightEnvmap():Light(),m_scale{1.0f},m_bitmap{nullptr}
{
  
}
