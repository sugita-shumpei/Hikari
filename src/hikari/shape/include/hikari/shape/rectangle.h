#pragma once
#include <hikari/core/shape.h>
namespace hikari {
  struct ShapeRectangle : public Shape {
    static constexpr Uuid ID() { return Uuid::from_string("55352985-23E7-4F0B-A3D2-FCE2AC6DE8E7").value(); }
    static auto create() -> std::shared_ptr<ShapeRectangle>;
    virtual ~ShapeRectangle() noexcept;
    Uuid getID() const override;
  protected:
    ShapeRectangle();
  };
}
