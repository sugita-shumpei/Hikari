#pragma once
#include <hikari/core/light.h>
#include <hikari/core/shape.h>
#include <hikari/core/spectrum.h>
namespace hikari {
  struct Node;
  struct LightConstant : public Light {
    static constexpr Uuid ID() { return Uuid::from_string("19EFF818-4F3D-466D-98AE-3C5BDA2F878F").value(); }
    static auto create() -> std::shared_ptr<LightConstant>;
    virtual ~LightConstant();

    Uuid getID() const override;

    void setRadiance(const SpectrumPtr& radiance);
    auto getRadiance()const->SpectrumPtr;
  private:
    LightConstant();
  private:
    SpectrumPtr m_radiance;
  };
}
