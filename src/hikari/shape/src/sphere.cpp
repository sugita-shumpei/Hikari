#include <hikari/shape/sphere.h>
auto hikari::ShapeSphere::create(const Vec3& center, F32 radius) -> std::shared_ptr<ShapeSphere> {
  return std::shared_ptr<ShapeSphere>(new ShapeSphere(center,radius));
}
hikari::ShapeSphere::~ShapeSphere() noexcept {}
void hikari::ShapeSphere::setCenter(const Vec3& center) { m_center = center; }
void hikari::ShapeSphere::setRadius(F32         radius) { m_radius = radius; }
auto hikari::ShapeSphere::getCenter() const->hikari::Vec3 { return m_center; }
auto hikari::ShapeSphere::getRadius() const->hikari::F32  { return m_radius;}
hikari::ShapeSphere::ShapeSphere(const Vec3& center, F32 radius) : Shape(), m_center{ center }, m_radius{ radius } {}

hikari::Uuid hikari::ShapeSphere::getID() const
{
  return ID();
}
