#pragma once
#include <hikari/shape/mesh.h>
#include "serialized_data.h"
namespace hikari {
  struct ShapeMitsubaSerializedMesh : public ShapeMesh {
    static constexpr Uuid ID() { return Uuid::from_string("9407CB95-8246-4A3A-BAF1-6E1E71FA26A3").value(); }

    static auto create(
      MitsubaSerializedDataManager&     manager ,
      const String&                     filename,
      U32                               shape_idx = 0
    ) -> std::shared_ptr<ShapeMitsubaSerializedMesh>;
    virtual ~ShapeMitsubaSerializedMesh();

    auto getFilename() const -> String;
    auto getShapeIndex()const-> U32;

    Uuid getID() const override;
    auto getVertexCount() const->U32 override;
    auto getFaceCount() const->U32 override;
    void clear() override;
    auto getVertexPositions() const -> const std::vector<Vec3> & override;
    auto getVertexNormals() const -> const std::vector<Vec3> & override;
    auto getVertexBinormals() const -> const std::vector<Vec4> & override;
    auto getVertexUVs() const -> const std::vector<Vec2> & override;
    auto getVertexColors() const -> const std::vector<Vec3> & override;
    auto getFaces() const -> const std::vector<U32> & override;
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
    ShapeMitsubaSerializedMesh(const std::shared_ptr<ShapeMesh>& mesh, const String& filename, U32 shape_index);
    String                     m_filename;
    U32                        m_shape_index;
    std::shared_ptr<ShapeMesh> m_mesh;
  };
}
