#pragma once
#include <hikari/core/shape.h>
namespace hikari {
  struct ShapeCube : public Shape {
    static constexpr Uuid ID() { return Uuid::from_string("B0472404-6F23-4B47-B2DF-6929FAA61C20").value(); }
    static auto create() -> std::shared_ptr<ShapeCube>;
    virtual ~ShapeCube() noexcept;
    Uuid getID() const override;
  protected:
    ShapeCube();

  };
}
