#pragma once
#include <hikari/core/shape.h>
#include <hikari/core/data_type.h>
namespace hikari {
  struct ShapeTriangle : public hikari::Shape {
    // [Guid("441C7D69-F2D9-486E-AE3E-94C674B1CF6E")]
    static constexpr Uuid ID() { return Uuid::from_string("441C7D69-F2D9-486E-AE3E-94C674B1CF6E").value(); }
    static auto create(const Vec3& v0 = Vec3(0.0f,0.0f,0.0f), const Vec3& v1 = Vec3(1.0f, 0.0f, 0.0f), const Vec3& v2 = Vec3(0.0f, 1.0f, 0.0f)) -> std::shared_ptr<ShapeTriangle>;
    virtual ~ShapeTriangle() noexcept;

    Uuid getID() const override;

    void setVertex0(const Vec3& v0);
    void setVertex1(const Vec3& v1);
    void setVertex2(const Vec3& v2);

    auto getVertex0() const->Vec3;
    auto getVertex1() const->Vec3;
    auto getVertex2() const->Vec3;
  private:
    ShapeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2);
  private:
    Vec3 m_vertex_0;
    Vec3 m_vertex_1;
    Vec3 m_vertex_2;
  };
}
