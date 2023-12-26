#include <hikari/core/surface.h>

auto hikari::Surface::create() -> std::shared_ptr<Surface>
{
    return std::shared_ptr<Surface>(new Surface());
}

hikari::Surface::~Surface() noexcept
{
}

void hikari::Surface::setOpacity(const SpectrumOrTexture& opacity)
{
  m_opacity = opacity;
}

auto hikari::Surface::getOpacity() -> SpectrumOrTexture
{
    return m_opacity;
}

auto hikari::Surface::getSubSurface() -> std::shared_ptr<SubSurface>
{
    return m_subsurfaces[0];
}

void hikari::Surface::setSubSurface(const std::shared_ptr<SubSurface>& subsurface)
{
  m_subsurfaces[0] = subsurface;
  m_subsurfaces[1] = nullptr;
}

auto hikari::Surface::getSubSurface(U32 idx) -> std::shared_ptr<SubSurface>
{
  if (idx >= 2) { return nullptr; }
  return m_subsurfaces[idx];
}

void hikari::Surface::setSubSurface(U32 idx, const std::shared_ptr<SubSurface>& subsurface)
{
  if (!subsurface) { return; }
  if (idx >= 2) { return; }
  if (idx == 0) {
    if (!subsurface) {
      m_subsurfaces[0] = nullptr;
      m_subsurfaces[1] = nullptr;
    }
    else {
      m_subsurfaces[0] = subsurface;
    }
    return;
  }
  if (idx == 1) {
    if (!m_subsurfaces[0]) {
      return;
    }
    else {
      m_subsurfaces[1] = subsurface;
    }
  }
}

hikari::Bool hikari::Surface::isTwoSided() const
{
    return m_subsurfaces[1]!=nullptr;
}

hikari::Surface::Surface():m_opacity{}, m_subsurfaces{nullptr,nullptr}
{
}
