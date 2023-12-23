#pragma once
#include <hikari/core/light.h>
#include <hikari/core/shape.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct Node;
  struct LightArea : public Light {
    static constexpr Uuid ID() { return Uuid::from_string("8196EBD8-1DD1-4605-9B4A-02B843E48A77").value(); }
    static auto create() -> std::shared_ptr<LightArea>;
    virtual ~LightArea();

    auto getShape() -> std::shared_ptr<Shape>;
    Uuid getID() const override;

    void setRadiance(const SpectrumOrTexture& radiance);
    auto getRadiance() const->SpectrumOrTexture;
  private:
    LightArea();
  private:
    SpectrumOrTexture m_radiance;
  };
}
