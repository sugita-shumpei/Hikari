#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/variant.h>
#include <memory>
#include <array>
namespace hikari
{
  struct SubSurface;
  struct Texture;
  struct Surface : public std::enable_shared_from_this<Surface> {
    static auto create() -> std::shared_ptr<Surface>;
    virtual ~Surface() noexcept;

    void setOpacity(const SpectrumOrTexture& opacity);
    auto getOpacity() -> SpectrumOrTexture;

    auto getSubSurface() -> std::shared_ptr<SubSurface>;
    void setSubSurface(const std::shared_ptr<SubSurface>& subsurface);

    auto getSubSurface(U32 idx) -> std::shared_ptr<SubSurface>;
    void setSubSurface(U32 idx, const std::shared_ptr<SubSurface>& subsurface);

    Bool isTwoSided() const;
  private:
    Surface();
    SpectrumOrTexture m_opacity;
    std::array<std::shared_ptr<SubSurface>, 2 > m_subsurfaces;
  };
  using SurfacePtr = std::shared_ptr<Surface>;
}
