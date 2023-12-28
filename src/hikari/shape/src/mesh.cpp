#include <hikari/shape/mesh.h>

namespace hikari {
  struct ShapeMeshImpl : public ShapeMesh {
    ShapeMeshImpl() : ShapeMesh() {}
    virtual ~ShapeMeshImpl() noexcept;

    auto getVertexCount() const->U32 override;
    auto getFaceCount()   const->U32 override;

    void clear() override;

    auto getVertexPositions() const -> std::vector<Vec3> override;
    auto getVertexNormals()   const -> std::vector<Vec3> override;
    auto getVertexBinormals() const -> std::vector<Vec4> override;
    auto getVertexUVs()       const -> std::vector<Vec2> override;
    auto getVertexColors()    const -> std::vector<Vec3> override;
    auto getFaces()           const -> std::vector<U32>  override;

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
  struct ShapeMeshInstance : public ShapeMesh 
  {
    ShapeMeshInstance(const std::shared_ptr<ShapeMesh>& base);
    virtual ~ShapeMeshInstance();

    auto getVertexCount() const->U32 override;
    auto getFaceCount() const->U32 override;
    void clear() override;
    auto getVertexPositions() const ->  std::vector<Vec3>  override;
    auto getVertexNormals() const ->  std::vector<Vec3>  override;
    auto getVertexBinormals() const ->  std::vector<Vec4>  override;
    auto getVertexUVs() const ->  std::vector<Vec2>  override;
    auto getVertexColors() const ->  std::vector<Vec3>  override;
    auto getFaces() const -> std::vector<U32> override;
    void setVertexPositions(const std::vector<Vec3>& vertex_positions) override;
    void setVertexNormals(const std::vector<Vec3>& vertex_normals) override;
    void setVertexBinormals(const std::vector<Vec4>& vertex_binormals) override;
    void setVertexUVs(const std::vector<Vec2>& vertex_uvs) override;
    void setVertexColors(const std::vector<Vec3>& vertex_colors) override;
    void setFaces(const std::vector<U32>& faces) override;
    Bool getFlipUVs() const override;
    Bool getFaceNormals() const override;
    void setFlipUVs(Bool flip_uvs) override;
    void setFaceNormals(Bool face_normals) override;
    Bool hasVertexNormals() const override;
    Bool hasVertexBinormals() const override;
    Bool hasVertexUVs() const override;
    Bool hasVertexColors() const override;
  private:
    std::shared_ptr<ShapeMesh> m_base;
    Bool                       m_flip_uvs     = false;
    Bool                       m_face_normals = false;
  };
}

auto hikari::ShapeMesh::create() -> std::shared_ptr<ShapeMesh>
{
  return std::shared_ptr<ShapeMesh>(new ShapeMeshImpl());
}

auto hikari::ShapeMesh::makeInstance(const std::shared_ptr<ShapeMesh>& shape) -> std::shared_ptr<ShapeMesh>
{
  if (!shape) { return nullptr; }
  return std::shared_ptr<ShapeMesh>(new ShapeMeshInstance(shape));
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

auto hikari::ShapeMeshImpl::getVertexPositions() const -> std::vector<Vec3> { return m_vertex_positions; }

auto hikari::ShapeMeshImpl::getVertexNormals() const -> std::vector<Vec3> { return m_vertex_normals; }

auto hikari::ShapeMeshImpl::getVertexBinormals() const -> std::vector<Vec4> { return m_vertex_binormals; }

auto hikari::ShapeMeshImpl::getVertexUVs() const -> std::vector<Vec2> { return m_vertex_uvs; }

auto hikari::ShapeMeshImpl::getVertexColors() const -> std::vector<Vec3>{ return m_vertex_colors;}

auto hikari::ShapeMeshImpl::getFaces() const -> std::vector<U32> { return m_faces; }

void hikari::ShapeMeshImpl::setVertexPositions(const std::vector<Vec3>& vertex_positions) {
  if (m_vertex_count != vertex_positions.size()) {
    clear();
    m_vertex_count = vertex_positions.size();
    m_vertex_positions.resize(m_vertex_count);
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

hikari::ShapeMeshInstance::~ShapeMeshInstance()
{
}


hikari::ShapeMeshInstance::ShapeMeshInstance(const std::shared_ptr<ShapeMesh>& base):
  ShapeMesh(),
  m_base{ base },
  m_face_normals {false},
  m_flip_uvs{false}
{
  m_face_normals = base->getFaceNormals();
  m_flip_uvs     = base->getFlipUVs();
}

auto hikari::ShapeMeshInstance::getVertexCount() const -> U32
{
  return m_base->getVertexCount();
}

auto hikari::ShapeMeshInstance::getFaceCount() const -> U32
{
  return m_base->getFaceCount();
}

void hikari::ShapeMeshInstance::clear()
{
  return;
}

auto hikari::ShapeMeshInstance::getVertexPositions() const -> std::vector<Vec3>
{
  return m_base->getVertexPositions();
}

auto hikari::ShapeMeshInstance::getVertexNormals() const -> std::vector<Vec3>
{
  return m_base->getVertexNormals();
}

auto hikari::ShapeMeshInstance::getVertexBinormals() const -> std::vector<Vec4>
{
  return m_base->getVertexBinormals();
}

auto hikari::ShapeMeshInstance::getVertexUVs() const -> std::vector<Vec2>
{
  return m_base->getVertexUVs();
}

auto hikari::ShapeMeshInstance::getVertexColors() const -> std::vector<Vec3>
{
  return m_base->getVertexColors();
}

auto hikari::ShapeMeshInstance::getFaces() const -> std::vector<U32>
{
  return m_base->getFaces();
}

void hikari::ShapeMeshInstance::setVertexPositions(const std::vector<Vec3>& vertex_positions)
{
  return;
}

void hikari::ShapeMeshInstance::setVertexNormals(const std::vector<Vec3>& vertex_normals)
{
  return;
}

void hikari::ShapeMeshInstance::setVertexBinormals(const std::vector<Vec4>& vertex_binormals)
{
  return;
}

void hikari::ShapeMeshInstance::setVertexUVs(const std::vector<Vec2>& vertex_uvs)
{
  return;
}

void hikari::ShapeMeshInstance::setVertexColors(const std::vector<Vec3>& vertex_colors)
{
  return;
}

void hikari::ShapeMeshInstance::setFaces(const std::vector<U32>& faces)
{
  return;
}

hikari::Bool hikari::ShapeMeshInstance::getFlipUVs() const
{
  return m_flip_uvs;
}

hikari::Bool hikari::ShapeMeshInstance::getFaceNormals() const
{
  return m_face_normals;
}

void hikari::ShapeMeshInstance::setFlipUVs(Bool flip_uvs)
{
  m_flip_uvs = flip_uvs;
}

void hikari::ShapeMeshInstance::setFaceNormals(Bool face_normals)
{
  m_face_normals = face_normals;
}

hikari::Bool hikari::ShapeMeshInstance::hasVertexNormals() const
{
  return m_base->hasVertexNormals();
}

hikari::Bool hikari::ShapeMeshInstance::hasVertexBinormals() const
{
  return m_base->hasVertexBinormals();
}

hikari::Bool hikari::ShapeMeshInstance::hasVertexUVs() const
{
  return m_base->hasVertexUVs();
}

hikari::Bool hikari::ShapeMeshInstance::hasVertexColors() const
{
  return m_base->hasVertexColors();
}


