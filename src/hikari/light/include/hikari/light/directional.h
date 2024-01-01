#pragma once
#include <hikari/core/light.h>
#include <hikari/core/shape.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct Node;
  struct LightDirectional : public Light {
    static constexpr Uuid ID() { return Uuid::from_string("C0E670D2-0218-43AE-A475-19F1D7A42E38").value(); }
    static auto create() -> std::shared_ptr<LightDirectional>;
    virtual ~LightDirectional();

    Uuid getID() const override;

    void setIrradiance(const SpectrumPtr& radiance);
    auto getIrradiance() const->SpectrumPtr;

    void setDirection(const Vec3& direction);
    auto getDirection() const->std::optional<Vec3>;
  private:
    LightDirectional();
  private:
    SpectrumPtr         m_irradiance;
    std::optional<Vec3> m_direction;
  };
}
