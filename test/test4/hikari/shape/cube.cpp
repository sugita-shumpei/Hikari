#include <hikari/shape/cube.h>

auto hikari::shape::ShapeCubeObject::create(const Str& name) -> std::shared_ptr<ShapeCubeObject>
{
  // hikari::Vec3{ -1.0f,-1.0f,0.0f },
  // hikari::Vec3{ +1.0f,-1.0f,0.0f },
  // hikari::Vec3{ +1.0f,+1.0f,0.0f },
  // hikari::Vec3{ -1.0f,+1.0f,0.0f }
  //  {0,1,2, 2,3,0}; +Z {0,2,1,2,0,3}; -Z
  // EY EZ EX
  // EZ EX EY
  // EX EY EZ
  std::array<std::array<hikari::F32, 3>, 4>  base_vertices = {
    std::array<hikari::F32,3>{-1.0f,-1.0f,0.0f},
    std::array<hikari::F32,3>{+1.0f,-1.0f,0.0f},
    std::array<hikari::F32,3>{+1.0f,+1.0f,0.0f},
    std::array<hikari::F32,3>{-1.0f,+1.0f,0.0f}
  };
  std::array<std::array<hikari::F32, 2>, 4>  base_uvs = {
    std::array<hikari::F32, 2>{ 0.0f, 0.0f},
    std::array<hikari::F32, 2>{+1.0f, 0.0f},
    std::array<hikari::F32, 2>{+1.0f,+1.0f},
    std::array<hikari::F32, 2>{ 0.0f,+1.0f}
  };

  std::vector<hikari::Vec3> vertices = {}; vertices.resize(24);
  std::vector<hikari::Vec3> normals = {}; normals.resize(24);
  std::vector<hikari::Vec2> uvs = {}; uvs.resize(24);
  std::vector<hikari::U32> faces = {}; faces.resize(36);

  {
    size_t vertex_off = 0;
    size_t index_off = 0;
    for (int i = 0; i < 3; ++i) {
      auto ax = (i + 1) % 3;
      auto ay = (i + 2) % 3;
      auto az = (i + 3) % 3;
      faces[index_off + 0] = vertex_off + 0; faces[index_off + 1] = vertex_off + 1; faces[index_off + 2] = vertex_off + 2;
      faces[index_off + 3] = vertex_off + 2; faces[index_off + 4] = vertex_off + 3; faces[index_off + 5] = vertex_off + 0;
      faces[index_off + 6] = vertex_off + 4; faces[index_off + 7] = vertex_off + 5; faces[index_off + 8] = vertex_off + 6;
      faces[index_off + 9] = vertex_off + 6; faces[index_off + 10] = vertex_off + 7; faces[index_off + 11] = vertex_off + 4;
      index_off += 12;
      // positiveの場合
      for (int j = 0; j < 4; ++j) {
        vertices[vertex_off + j][ax] = base_vertices[j][0];
        vertices[vertex_off + j][ay] = base_vertices[j][1];
        vertices[vertex_off + j][az] = -1.0f;
        normals[vertex_off + j][ax] = 0.0f;
        normals[vertex_off + j][ay] = 0.0f;
        normals[vertex_off + j][az] = -1.0f;
        uvs[vertex_off + j][0] = base_uvs[j][0];
        uvs[vertex_off + j][1] = base_uvs[j][1];
      }
      vertex_off += 4;
      // negativeの場合
      for (int j = 0; j < 4; ++j) {
        vertices[vertex_off + j][ax] = base_vertices[j][0];
        vertices[vertex_off + j][ay] = base_vertices[j][1];
        vertices[vertex_off + j][az] = +1.0f;
        normals[vertex_off + j][ax] = 0.0f;
        normals[vertex_off + j][ay] = 0.0f;
        normals[vertex_off + j][az] = +1.0f;
        uvs[vertex_off + j][0] = base_uvs[j][0];
        uvs[vertex_off + j][1] = base_uvs[j][1];
      }
      vertex_off += 4;
    }
  }

  auto res = new ShapeCubeObject(name);
  res->setPositions(vertices);
  res->setNormals(normals);
  res->setUV(uvs);
  res->setIndices(faces);
  res->toImmutable();
  return std::shared_ptr<ShapeCubeObject>(res);
}

auto hikari::shape::ShapeCubeSerializer::getTypeString() const noexcept -> Str
{
  return ShapeCubeObject::TypeString();
}

auto hikari::shape::ShapeCubeSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto shape_rect = ObjectUtils::convert<ShapeCubeObject>(object);
  if (!shape_rect) { return Json(); }
  Json json;
  json["type"] = "ShapeCube";
  json["name"] = shape_rect->getName();
  json["properties"] = {};
  json["properties"]["flip_normals"] = shape_rect->getFlipNormals();
  return json;
}

auto hikari::shape::ShapeCubeDeserializer::getTypeString() const noexcept -> Str
{
  return ShapeCubeObject::TypeString();
}

auto hikari::shape::ShapeCubeDeserializer::eval(const Json& json) const -> std::shared_ptr<Object>
{
  ;
  auto iter_name = json.find("name");
  if (iter_name == json.end()) { return nullptr; }
  if (!iter_name.value().is_string()) { return nullptr; }
  auto shape_rect = ShapeCubeObject::create(iter_name.value().get<Str>());
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
