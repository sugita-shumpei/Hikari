#pragma once
#include <hikari/core/variant.h>
#include <array>
#include <memory>
namespace hikari {
  struct Texture;
  struct Node;
  struct Bsdf;
  struct Medium;
  struct Material : public std::enable_shared_from_this<Material> {
    static auto create() -> std::shared_ptr<Material>;
    virtual ~Material() noexcept;
    auto getBsdf() -> std::shared_ptr<Bsdf>;
    auto getInternalMedium() ->  std::shared_ptr<Medium>;
    auto getExternalMedium() ->  std::shared_ptr<Medium>;

    void setBsdf(const std::shared_ptr<Bsdf>& bsdf);
    void setInternalMedium(const std::shared_ptr<Medium>& medium);
    void setExternalMedium(const std::shared_ptr<Medium>& medium);
  protected:
    Material();
  private:
    std::shared_ptr<Bsdf>                  m_bsdf;
    std::shared_ptr<Medium>                m_internal_medium;
    std::shared_ptr<Medium>                m_external_medium;
  };
  using  MaterialPtr = std::shared_ptr<Material>;
}
