#include <hikari/shape/rect.h>

auto hikari::shape::ShapeRectObject::create(const Str& name) -> std::shared_ptr<ShapeRectObject>
{
  std::vector<hikari::Vec3> vertices = {
hikari::Vec3{-1.0f,-1.0f,0.0f},
hikari::Vec3{+1.0f,-1.0f,0.0f},
hikari::Vec3{+1.0f,+1.0f,0.0f},
hikari::Vec3{-1.0f,+1.0f,0.0f}
  };
  std::vector<hikari::Vec3> normals = {
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f}
  };
  std::vector<hikari::Vec2> uvs = {
    hikari::Vec2{ 0.0f, 0.0f},
    hikari::Vec2{+1.0f, 0.0f},
    hikari::Vec2{+1.0f,+1.0f},
    hikari::Vec2{ 0.0f,+1.0f}
  };
  std::vector<hikari::U32>  faces = { 0,1,2, 2,3,0 };

  auto res = new ShapeRectObject(name);
  res->setPositions(vertices);
  res->setNormals(normals);
  res->setUV(uvs);
  res->setIndices(faces);
  res->toImmutable();
  return std::shared_ptr<ShapeRectObject>(res);
}

auto hikari::shape::ShapeRectSerializer::getTypeString() const noexcept -> Str
{
  return ShapeRectObject::TypeString();
}

auto hikari::shape::ShapeRectSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto shape_rect = ObjectUtils::convert<ShapeRectObject>(object);
  if (!shape_rect) { return Json(); }
  Json json;
  json["type"] = "ShapeRect";
  json["name"] = shape_rect->getName();
  json["properties"] = {};
  json["properties"]["flip_normals"] = shape_rect->getFlipNormals();
  return json;
}

auto hikari::shape::ShapeRectDeserializer::getTypeString() const noexcept -> Str
{
  return ShapeRectObject::TypeString();
}

auto hikari::shape::ShapeRectDeserializer::eval(const Json& json) const -> std::shared_ptr<Object>
{;
  auto iter_name = json.find("name");
  if (iter_name == json.end()) { return nullptr; }
  if (!iter_name.value().is_string()) { return nullptr; }
  auto shape_rect = ShapeRectObject::create(iter_name.value().get<Str>());
  auto iter_properties = json.find("properties");
  if (iter_properties == json.end()) { return shape_rect; }
  auto& properties = iter_properties.value();
  try {
    auto flip_normals = properties.find("flip_normals");
    if (flip_normals != properties.end()) {
      shape_rect->setFlipNormals(flip_normals.value().get<Bool>());
    }
  }
  catch (...) {

  }
  return shape_rect;
}
