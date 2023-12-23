#pragma once
#include <hikari/core/shape.h>
#include <vector>
namespace hikari {
  struct ShapeMesh : public Shape {
    static constexpr Uuid ID() {  return Uuid::from_string("F127AAE2-4240-4910-A187-BBB1588B7FC4").value();  }

    static auto create() -> std::shared_ptr<ShapeMesh> ;
    virtual ~ShapeMesh() noexcept {}

    virtual auto getVertexCount() const -> U32 =0;
    virtual auto getFaceCount()   const -> U32 =0;

    virtual void clear() =0;

    virtual auto getVertexPositions() const -> const std::vector<Vec3>& =0;
    virtual auto getVertexNormals()   const -> const std::vector<Vec3>& =0;
    virtual auto getVertexBinormals() const -> const std::vector<Vec4>& =0;
    virtual auto getVertexUVs()       const -> const std::vector<Vec2>& =0;
    virtual auto getVertexColors()    const -> const std::vector<Vec3>& =0;
    virtual auto getFaces()           const -> const std::vector<U32> & =0;

    virtual void setVertexPositions(const std::vector<Vec3>& vertex_positions) =0;
    virtual void setVertexNormals  (const std::vector<Vec3>& vertex_normals  ) =0;
    virtual void setVertexBinormals(const std::vector<Vec4>& vertex_binormals) =0;
    virtual void setVertexUVs      (const std::vector<Vec2>& vertex_uvs      ) =0;
    virtual void setVertexColors   (const std::vector<Vec3>& vertex_colors   ) =0;
    virtual void setFaces          (const std::vector<U32> & faces           ) =0;

    virtual Bool getFlipUVs()        const =0;
    virtual Bool getFaceNormals()    const =0;

    virtual void setFlipUVs(Bool flip_uvs) =0;
    virtual void setFaceNormals(Bool face_normals) =0;

    virtual Bool hasVertexNormals()  const =0;
    virtual Bool hasVertexBinormals()const =0;
    virtual Bool hasVertexUVs()      const =0;
    virtual Bool hasVertexColors()   const =0;

    Uuid getID() const override { return ID(); }
  protected:
    ShapeMesh() : Shape() {}
  };
}
