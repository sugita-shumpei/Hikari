#pragma once
#include <hikari/core/data_type.h>
#include <hikari/core/texture.h>
#include <memory>
#include <array>
namespace hikari
{
  struct Bsdf;
  struct SubSurface : public std::enable_shared_from_this<SubSurface> {
    static auto create() -> std::shared_ptr<SubSurface>;
    virtual ~SubSurface() noexcept;

    void setBsdf(const std::shared_ptr<Bsdf>& bsdf);
    auto getBsdf() -> std::shared_ptr<Bsdf>;

    auto getBumpMap()  -> TexturePtr;
    void setBumpMap(const TexturePtr& texture);

    auto getNormalMap()  -> TexturePtr;
    void setNormalMap(const TexturePtr& texture);

    auto getBumpScale() const->F32;
    void setBumpScale(F32 scale);
  private:
    SubSurface();
    std::shared_ptr<Bsdf> m_bsdf;
    TexturePtr            m_normal_map;
    TexturePtr            m_bump_map;
    F32                   m_bump_scale;
  };
  using SubSurfacePtr = std::shared_ptr<SubSurface>;
}
