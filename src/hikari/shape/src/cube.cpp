#include <hikari/shape/cube.h>

auto hikari::ShapeCube::create() -> std::shared_ptr<ShapeCube>
{
    return std::shared_ptr<ShapeCube>(new ShapeCube());
}

hikari::ShapeCube::~ShapeCube() noexcept
{
}

hikari::ShapeCube::ShapeCube() : Shape()
{
}

hikari::Uuid hikari::ShapeCube::getID() const
{
  return ID();
}
