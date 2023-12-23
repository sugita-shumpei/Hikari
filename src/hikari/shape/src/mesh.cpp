#include <hikari/shape/mesh.h>

namespace hikari {
  struct ShapeMeshImpl : public ShapeMesh {
    ShapeMeshImpl() : ShapeMesh() {}
    virtual ~ShapeMeshImpl() noexcept;

    auto getVertexCount() const->U32 override;
    auto getFaceCount()   const->U32 override;

    void clear() override;

    auto getVertexPositions() const -> const std::vector<Vec3>& override;
    auto getVertexNormals()   const -> const std::vector<Vec3>& override;
    auto getVertexBinormals() const -> const std::vector<Vec4>& override;
    auto getVertexUVs()       const -> const std::vector<Vec2>& override;
    auto getVertexColors()    const -> const std::vector<Vec3>& override;
    auto getFaces()           const -> const std::vector<U32>& override;

    void setVertexPositions(const std::vector<Vec3>& vertex_positions) override;
    void setVertexNormals(const std::vector<Vec3>& vertex_normals) override;
    void setVertexBinormals(const std::vector<Vec4>& vertex_binormals) override;
    void setVertexUVs(const std::vector<Vec2>& vertex_uvs) override;
    void setVertexColors(const std::vector<Vec3>& vertex_colors) override;
    void setFaces(const std::vector<U32>& faces) override;

    Bool getFlipUVs()        const override;
    Bool getFaceNormals()    const override;

    void setFlipUVs(Bool flip_uvs) override;
    void setFaceNormals(Bool face_normals) override;

    Bool hasVertexNormals()  const override;
    Bool hasVertexBinormals()const override;
    Bool hasVertexUVs()      const override;
    Bool hasVertexColors()   const override;

  private:
    std::vector<Vec3> m_vertex_positions = {};
    std::vector<Vec3> m_vertex_normals = {};
    std::vector<Vec4> m_vertex_binormals = {};
    std::vector<Vec2> m_vertex_uvs = {};
    std::vector<Vec3> m_vertex_colors = {};
    std::vector<U32>  m_faces = {};
    U32               m_vertex_count = 0;
    U32               m_face_count = 0;
    Bool              m_flip_uvs = false;
    Bool              m_face_normals = false;

  };
}

auto hikari::ShapeMesh::create() -> std::shared_ptr<ShapeMesh>
{
  return std::shared_ptr<ShapeMesh>(new ShapeMeshImpl());
}

hikari::ShapeMeshImpl::~ShapeMeshImpl() noexcept
{
}

auto hikari::ShapeMeshImpl::getVertexCount() const -> U32 { return m_vertex_count; }

auto hikari::ShapeMeshImpl::getFaceCount() const -> U32 { return m_face_count; }

void hikari::ShapeMeshImpl::clear()
{
  m_vertex_positions.clear();
  m_vertex_normals.clear();
  m_vertex_binormals.clear();
  m_vertex_uvs.clear();
  m_faces.clear();
  m_vertex_count = 0;
  m_face_count = 0;
}

auto hikari::ShapeMeshImpl::getVertexPositions() const -> const std::vector<Vec3>& { return m_vertex_positions; }

auto hikari::ShapeMeshImpl::getVertexNormals() const -> const std::vector<Vec3>& { return m_vertex_normals; }

auto hikari::ShapeMeshImpl::getVertexBinormals() const -> const std::vector<Vec4>& { return m_vertex_binormals; }

auto hikari::ShapeMeshImpl::getVertexUVs() const -> const std::vector<Vec2>& { return m_vertex_uvs; }

auto hikari::ShapeMeshImpl::getVertexColors() const -> const std::vector<Vec3>&{ return m_vertex_colors;}

auto hikari::ShapeMeshImpl::getFaces() const -> const std::vector<U32>& { return m_faces; }

void hikari::ShapeMeshImpl::setVertexPositions(const std::vector<Vec3>& vertex_positions) {
  if (m_vertex_count != vertex_positions.size()) {
    clear();
    m_vertex_count = vertex_positions.size();
  }
  for (auto i = size_t(0); i < m_vertex_count; ++i) {
    m_vertex_positions[i] = vertex_positions[i];
  }
}

void hikari::ShapeMeshImpl::setVertexNormals(const std::vector<Vec3>& vertex_normals) {
  if (m_vertex_count != vertex_normals.size()) { return; }
  m_vertex_normals = vertex_normals;
}

void hikari::ShapeMeshImpl::setVertexBinormals(const std::vector<Vec4>& vertex_binormals) {
  if (m_vertex_count != vertex_binormals.size()) { return; }
  m_vertex_binormals = vertex_binormals;
}

void hikari::ShapeMeshImpl::setVertexUVs(const std::vector<Vec2>& vertex_uvs) {
  if (m_vertex_count != vertex_uvs.size()) { return; }
  m_vertex_uvs = vertex_uvs;
}

void hikari::ShapeMeshImpl::setVertexColors(const std::vector<Vec3>& vertex_colors)
{
  if (m_vertex_count != vertex_colors.size()) { return; }
  m_vertex_colors = vertex_colors;
}

void hikari::ShapeMeshImpl::setFaces(const std::vector<U32>& faces) {
  if (faces.size() % 3 != 0) { return; }
  m_face_count = faces.size() / 3;
  m_faces = faces;
}

hikari::Bool hikari::ShapeMeshImpl::getFlipUVs() const { return m_flip_uvs; }

hikari::Bool hikari::ShapeMeshImpl::getFaceNormals() const { return m_face_normals; }

void hikari::ShapeMeshImpl::setFlipUVs(hikari::Bool flip_uvs) { m_flip_uvs = flip_uvs; }

void hikari::ShapeMeshImpl::setFaceNormals(hikari::Bool face_normals) { m_face_normals = face_normals; }

hikari::Bool hikari::ShapeMeshImpl::hasVertexNormals() const { return !m_vertex_normals.empty(); }

hikari::Bool hikari::ShapeMeshImpl::hasVertexBinormals() const { return !m_vertex_binormals.empty(); }

hikari::Bool hikari::ShapeMeshImpl::hasVertexUVs() const { return !m_vertex_uvs.empty(); }

hikari::Bool hikari::ShapeMeshImpl::hasVertexColors() const { return !m_vertex_colors.empty(); }

