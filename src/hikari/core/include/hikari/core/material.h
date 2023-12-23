#pragma once
#include <hikari/core/variant.h>
#include <memory>
namespace hikari {
  struct Texture;
  struct Node;
  struct Bsdf;
  struct Medium;
  struct Material : public std::enable_shared_from_this<Material> {
    static auto create() -> std::shared_ptr<Material>;
    virtual ~Material() noexcept;
    auto getBsdf()           ->  std::shared_ptr<Bsdf>  ;
    auto getInternalMedium() ->  std::shared_ptr<Medium>;
    auto getExternalMedium() ->  std::shared_ptr<Medium>;
    auto getOpacity()  const ->  SpectrumOrTexture;
    auto getBumpMap()        ->  TexturePtr;
    auto getBumpScale()const ->  F32;
    Bool getTwoSided()const;
    Bool hasBumpMap() const;
    Bool HasOpacity() const;

    void setBsdf(const std::shared_ptr<Bsdf>& bsdf);
    void setInternalMedium(const std::shared_ptr<Medium>& medium);
    void setExternalMedium(const std::shared_ptr<Medium>& medium);
    void setOpacity(const SpectrumOrTexture& opacity_map);
    void setBumpMap(const TexturePtr& bump_map);
    void setBumpScale(F32 bump_scale);
    void setTwoSided(Bool two_sided);
  protected:
    Material();
  private:
    std::shared_ptr<Bsdf>   m_bsdf;
    std::shared_ptr<Medium> m_internal_medium;
    std::shared_ptr<Medium> m_external_medium;
    SpectrumOrTexture       m_opacity;
    TexturePtr              m_bump_map;
    F32                     m_bump_scale;
    Bool                    m_two_sided ;
  };
  using MaterialPtr = std::shared_ptr<Material>;
}
