#pragma once
#include <hikari/core/shape.h>
namespace hikari {
  struct ShapeSphere : public Shape {
    static constexpr Uuid ID() { return Uuid::from_string("1505BE8A-7765-4353-B28A-0A22212329F4").value(); }
    static auto create(const Vec3& center = Vec3(0.0f, 0.0f, 0.0f), F32 radius = 1.0f) -> std::shared_ptr<ShapeSphere>;
    virtual ~ShapeSphere() noexcept;
    Uuid getID() const override;

    void setCenter(const Vec3& center);
    void setRadius(F32         radius);
    auto getCenter() const-> Vec3;
    auto getRadius() const-> F32;

    auto createMesh() -> std::shared_ptr<Shape>;
  private:
    ShapeSphere(const Vec3& center, F32 radius);
  private:
    Vec3 m_center;
    F32  m_radius;
  };
}
