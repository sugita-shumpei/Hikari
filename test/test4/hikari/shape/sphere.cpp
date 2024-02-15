#include <hikari/shape/sphere.h>

auto hikari::shape::ShapeSphereSerializer::getTypeString() const noexcept -> Str 
{
  return  hikari::shape::ShapeSphereObject::TypeString();
}

auto hikari::shape::ShapeSphereSerializer::eval(const std::shared_ptr<Object>& object) const -> Json 
{
  auto shape_sphere = ObjectUtils::convert<ShapeSphereObject>(object);
  if (!shape_sphere) { return Json(); }
  Json json;
  json["type"] = "ShapeSphere";
  json["name"] = shape_sphere->getName();
  json["properties"] = {};
  json["properties"]["center"] = serialize(shape_sphere->getCenter());
  json["properties"]["radius"] = shape_sphere->getRadius();
  json["properties"]["flip_normals"] = shape_sphere->getFlipNormals();
  return json;
}

auto hikari::shape::ShapeSphereDeserializer::getTypeString() const noexcept -> Str 
{
  return  hikari::shape::ShapeSphereObject::TypeString();
}

auto hikari::shape::ShapeSphereDeserializer::eval(const Json& json) const -> std::shared_ptr<Object> 
{
  auto iter_name = json.find("name");
  if (iter_name == json.end()) { return nullptr; }
  if (!iter_name.value().is_string()) { return nullptr; }
  auto shape_sphere = ShapeSphereObject::create(iter_name.value().get<Str>());
  auto iter_properties = json.find("properties");
  if (iter_properties == json.end()) { return shape_sphere; }
  auto& properties = iter_properties.value();
  auto iter_center       = properties.find("center");
  if (iter_center != properties.end()) { auto c = deserialize<Vec3>(iter_center.value()); if (c) { shape_sphere->setCenter(*c); } }
  auto iter_radius       = properties.find("radius");
  if (iter_radius != properties.end()) { auto r = deserialize<F32>(iter_radius.value()); if (r) { shape_sphere->setRadius(*r); } }
  auto iter_flip_normals = properties.find("flip_normals");
  if (iter_flip_normals != properties.end()) { auto f = deserialize<F32>(iter_flip_normals.value()); if (f) { shape_sphere->setFlipNormals(*f); } }
  return shape_sphere;
}
