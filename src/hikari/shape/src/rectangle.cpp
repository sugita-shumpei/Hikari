#include <hikari/shape/rectangle.h>

auto hikari::ShapeRectangle::create() -> std::shared_ptr<ShapeRectangle>
{
  return std::shared_ptr<ShapeRectangle>(new ShapeRectangle());
}

hikari::ShapeRectangle::~ShapeRectangle() noexcept
{
}

hikari::ShapeRectangle::ShapeRectangle():Shape()
{
}

hikari::Uuid hikari::ShapeRectangle::getID() const
{
  return ID();
}
