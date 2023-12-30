#include <hikari/shape/cube.h>
#include <hikari/shape/mesh.h>
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

auto hikari::ShapeCube::createMesh() -> std::shared_ptr<Shape>
{    // proceduralにcubeを作成する
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
  std::vector<hikari::Vec3> normals  = {}; normals.resize(24);
  std::vector<hikari::Vec2> uvs      = {}; uvs.resize(24);
  std::vector<hikari::U32> faces     = {}; faces.resize(36);

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

  auto res = ShapeMesh::create();
  res->setVertexPositions(vertices);
  res->setVertexNormals(normals);
  res->setVertexUVs(uvs);
  res->setFaces(faces);
  return res;
}
