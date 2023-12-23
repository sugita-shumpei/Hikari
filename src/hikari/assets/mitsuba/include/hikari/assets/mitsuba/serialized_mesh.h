#pragma once
#include <hikari/shape/mesh.h>
namespace hikari {
  struct ShapeMitsubaSerializedMesh : public Shape {
    static constexpr Uuid ID() { return Uuid::from_string("9407CB95-8246-4A3A-BAF1-6E1E71FA26A3").value(); }

    static auto create() -> std::shared_ptr<ShapeMitsubaSerializedMesh>;
    virtual ~ShapeMitsubaSerializedMesh();

    void setFilename(const String& filename);
    auto getFilename() const-> String;

    auto getShapeIndex()const->U32;
    void setShapeIndex(U32 shape_index);

    Uuid getID() const override;
  private:
    ShapeMitsubaSerializedMesh();
    String m_filename;
    U32    m_shape_index;
  };
}
