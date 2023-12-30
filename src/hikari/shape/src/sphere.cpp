#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
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
auto hikari::ShapeSphere::createMesh() -> std::shared_ptr<Shape>
{
  //___________________
  // |___|___|___|___|
  // |___|___|___|___|
  // |___|___|___|___|
  //_|___|___|___|___|_
  constexpr size_t div_count = 128;
  // 側面は(div_count-2)*div_countの台形
  // 上下はdivcountの三角形
  std::vector<hikari::Vec3> vertices ((2 * 3 * div_count) + (div_count * (div_count - 2) * 4));
  std::vector<hikari::Vec3> normals  ((2 * 3 * div_count) + (div_count * (div_count - 2) * 4));
  std::vector<hikari::Vec2> uvs      ((2 * 3 * div_count) + (div_count * (div_count - 2) * 4));
  std::vector<hikari::U32>  indices  ((2 * 3 * div_count) + (div_count * (div_count - 2) * 6));

  size_t vertex_offset = 0;
  size_t index_offset = 0;
  {
    auto cos_tht = std::cos(glm::pi<float>() / static_cast<float>(div_count));
    auto sin_tht = std::sin(glm::pi<float>() / static_cast<float>(div_count));
    for (size_t j = 0; j < div_count; ++j) {
      auto cos_phi_0 = std::cos((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto sin_phi_0 = std::sin((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto cos_phi_1 = std::cos((2.0f * glm::pi<float>() * ((j+1)%div_count)) / static_cast<float>(div_count));
      auto sin_phi_1 = std::sin((2.0f * glm::pi<float>() * ((j+1)%div_count)) / static_cast<float>(div_count));
      Vec3 v0 = { sin_tht * cos_phi_0,sin_tht * sin_phi_0, cos_tht };
      Vec3 v1 = { sin_tht * cos_phi_1,sin_tht * sin_phi_1, cos_tht };
      Vec3 v2 = { 0.0f,0.0f,1.0f };
      indices[index_offset +0]   = vertex_offset;
      indices[index_offset +1]   = vertex_offset +1;
      indices[index_offset +2]   = vertex_offset +2;

      vertices[vertex_offset + 0] = m_radius * v0 + m_center;
      vertices[vertex_offset + 1] = m_radius * v1 + m_center;
      vertices[vertex_offset + 2] = m_radius * v2 + m_center;
      auto vn                             = glm::normalize(glm::cross(v1 - v0, v2 - v0));
      normals[vertex_offset +0]  = vn;
      normals[vertex_offset +1]  = vn;
      normals[vertex_offset +2]  = vn;
      uvs[vertex_offset +0]      = { (j+0)/div_count   ,1.0f/div_count };
      uvs[vertex_offset +1]      = { (j+1)/div_count   ,1.0f/div_count };
      uvs[vertex_offset +2]      = { (j+0.5f)/div_count,0.0f };
      vertex_offset += 3;
      index_offset  += 3;
    }
  }
  for (size_t i = 1;i < div_count-1;++i){
    auto cos_tht_0 = std::cos((i*glm::pi<float>())/ static_cast<float>(div_count));
    auto sin_tht_0 = std::sin((i*glm::pi<float>())/ static_cast<float>(div_count));
    auto cos_tht_1 = std::cos(((i+1)*glm::pi<float>())/ static_cast<float>(div_count));
    auto sin_tht_1 = std::sin(((i+1)*glm::pi<float>())/ static_cast<float>(div_count));
    for (size_t j = 0; j < div_count; ++j) {
      auto cos_phi_0 = std::cos((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto sin_phi_0 = std::sin((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto cos_phi_1 = std::cos((2.0f * glm::pi<float>() * ((j + 1) % div_count)) / static_cast<float>(div_count));
      auto sin_phi_1 = std::sin((2.0f * glm::pi<float>() * ((j + 1) % div_count)) / static_cast<float>(div_count));
      Vec3 v0 = { sin_tht_1 * cos_phi_0,sin_tht_1 * sin_phi_0, cos_tht_1 };
      Vec3 v1 = { sin_tht_1 * cos_phi_1,sin_tht_1 * sin_phi_1, cos_tht_1 };
      Vec3 v2 = { sin_tht_0 * cos_phi_1,sin_tht_0 * sin_phi_1, cos_tht_0 };
      Vec3 v3 = { sin_tht_0 * cos_phi_0,sin_tht_0 * sin_phi_0, cos_tht_0 };

      indices[index_offset + 0] = vertex_offset;
      indices[index_offset + 1] = vertex_offset + 1;
      indices[index_offset + 2] = vertex_offset + 2;
      indices[index_offset + 3] = vertex_offset + 2;
      indices[index_offset + 4] = vertex_offset + 3;
      indices[index_offset + 5] = vertex_offset + 0;

      vertices[vertex_offset + 0] = m_radius * v0 + m_center;
      vertices[vertex_offset + 1] = m_radius * v1 + m_center;
      vertices[vertex_offset + 2] = m_radius * v2 + m_center;
      vertices[vertex_offset + 3] = m_radius * v3 + m_center;

      auto vn = glm::normalize(glm::cross(v1 - v0, v2 - v0));
      normals[vertex_offset + 0] = vn;
      normals[vertex_offset + 1] = vn;
      normals[vertex_offset + 2] = vn;
      normals[vertex_offset + 3] = vn;

      uvs[vertex_offset + 0] = { (j + 0) / div_count   ,(i + 1) / div_count };
      uvs[vertex_offset + 1] = { (j + 1) / div_count   ,(i + 1) / div_count };
      uvs[vertex_offset + 2] = { (j + 1) / div_count   ,(i + 0) / div_count };
      uvs[vertex_offset + 3] = { (j + 0) / div_count   ,(i + 0) / div_count };

      vertex_offset += 4;
      index_offset  += 6;
    }
  }
  {
    auto cos_tht =-std::cos(glm::pi<float>() / static_cast<float>(div_count));
    auto sin_tht = std::sin(glm::pi<float>() / static_cast<float>(div_count));
    for (size_t j = 0; j < div_count; ++j) {
      auto cos_phi_0 = std::cos((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto sin_phi_0 = std::sin((2.0f * glm::pi<float>() * j) / static_cast<float>(div_count));
      auto cos_phi_1 = std::cos((2.0f * glm::pi<float>() * ((j + 1) % div_count)) / static_cast<float>(div_count));
      auto sin_phi_1 = std::sin((2.0f * glm::pi<float>() * ((j + 1) % div_count)) / static_cast<float>(div_count));
      Vec3 v0 = { sin_tht * cos_phi_0,sin_tht * sin_phi_0, cos_tht };
      Vec3 v1 = { sin_tht * cos_phi_1,sin_tht * sin_phi_1, cos_tht };
      Vec3 v2 = { 0.0f,0.0f,-1.0f };
      indices[index_offset + 0] = vertex_offset + 1;
      indices[index_offset + 1] = vertex_offset + 0;
      indices[index_offset +  + 2] = vertex_offset + 2;

      vertices[vertex_offset + 0] = m_radius * v0 + m_center;
      vertices[vertex_offset + 1] = m_radius * v1 + m_center;
      vertices[vertex_offset + 2] = m_radius * v2 + m_center;
      auto vn =-glm::normalize(glm::cross(v1 - v0, v2 - v0));
      normals[vertex_offset + 0] = vn;
      normals[vertex_offset + 1] = vn;
      normals[vertex_offset + 2] = vn;
      uvs[vertex_offset + 0] = { (j + 0) / div_count   ,1.0f -(1.0f / div_count)};
      uvs[vertex_offset + 1] = { (j + 1) / div_count   ,1.0f -(1.0f / div_count)};
      uvs[vertex_offset + 2] = { (j + 0.5f) / div_count,1.0f };
      vertex_offset += 3;
      index_offset  += 3;
    }
  }


  auto res = ShapeMesh::create();
  res->setVertexPositions(vertices);
  res->setVertexNormals(normals);
  res->setVertexUVs(uvs);
  res->setFaces(indices);
  return res;
}
