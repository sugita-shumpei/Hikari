#include <hikari/core/subsurface.h>

auto hikari::SubSurface::create() -> std::shared_ptr<SubSurface>
{
    return std::shared_ptr<SubSurface>(new SubSurface());
}

hikari::SubSurface::~SubSurface() noexcept
{
}

void hikari::SubSurface::setBsdf(const std::shared_ptr<Bsdf>& bsdf)
{
  m_bsdf = bsdf;
}

auto hikari::SubSurface::getBsdf() -> std::shared_ptr<Bsdf>
{
    return m_bsdf;
}

auto hikari::SubSurface::getBumpMap() -> TexturePtr
{
    return m_bump_map;
}

void hikari::SubSurface::setBumpMap(const TexturePtr& texture)
{
  m_bump_map = texture;
}

auto hikari::SubSurface::getNormalMap() -> TexturePtr
{
    return m_normal_map;
}

void hikari::SubSurface::setNormalMap(const TexturePtr& texture)
{
  m_normal_map = texture;
}

auto hikari::SubSurface::getBumpScale() const -> F32
{
    return m_bump_scale;
}

void hikari::SubSurface::setBumpScale(F32 scale)
{
  m_bump_scale = scale;
}

hikari::SubSurface::SubSurface()
  :
  m_bsdf{nullptr},
  m_bump_map{nullptr},
  m_normal_map{nullptr},
  m_bump_scale{1.0f}
{
}
