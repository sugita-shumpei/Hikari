#include <hikari/shape/triangle.h>

auto hikari::ShapeTriangle::create(const Vec3& v0, const Vec3& v1, const Vec3& v2) -> std::shared_ptr<ShapeTriangle>
{
  return std::shared_ptr<ShapeTriangle>(new ShapeTriangle(v0,v1,v2));
}

hikari::ShapeTriangle::~ShapeTriangle() noexcept
{
}

void hikari::ShapeTriangle::setVertex0(const Vec3& v0)
{
  m_vertex_0 = v0;
}

void hikari::ShapeTriangle::setVertex1(const Vec3& v1)
{
  m_vertex_1 = v1;
}

void hikari::ShapeTriangle::setVertex2(const Vec3& v2)
{
  m_vertex_2 = v2;
}

auto hikari::ShapeTriangle::getVertex0() const -> Vec3
{
  return m_vertex_0;
}

auto hikari::ShapeTriangle::getVertex1() const -> Vec3
{
  return m_vertex_1;
}

auto hikari::ShapeTriangle::getVertex2() const -> Vec3
{
  return m_vertex_2;
}

hikari::ShapeTriangle::ShapeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2) : Shape(),
  m_vertex_0{v0},
  m_vertex_1{v1},
  m_vertex_2{v2}
{
}

hikari::Uuid hikari::ShapeTriangle::getID() const
{
  return ID();
}
