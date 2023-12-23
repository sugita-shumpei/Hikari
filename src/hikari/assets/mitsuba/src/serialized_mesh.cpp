#include <hikari/assets/mitsuba/serialized_mesh.h>

auto hikari::ShapeMitsubaSerializedMesh::create() -> std::shared_ptr<ShapeMitsubaSerializedMesh>
{
  return std::shared_ptr<ShapeMitsubaSerializedMesh>(new ShapeMitsubaSerializedMesh());
}

hikari::ShapeMitsubaSerializedMesh::~ShapeMitsubaSerializedMesh()
{
}

void hikari::ShapeMitsubaSerializedMesh::setFilename(const String& filename)
{
  m_filename = filename;
}

auto hikari::ShapeMitsubaSerializedMesh::getFilename() const -> String
{
  return m_filename;
}

auto hikari::ShapeMitsubaSerializedMesh::getShapeIndex() const -> U32
{
  return m_shape_index;
}

void hikari::ShapeMitsubaSerializedMesh::setShapeIndex(U32 shape_index)
{
  m_shape_index = shape_index;
}

hikari::Uuid hikari::ShapeMitsubaSerializedMesh::getID() const
{
    return ID();
}

hikari::ShapeMitsubaSerializedMesh::ShapeMitsubaSerializedMesh(): Shape(),m_shape_index{0}
{
}
