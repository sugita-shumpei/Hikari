#include <hikari/shape/rectangle.h>
#include <hikari/shape/mesh.h>
#include <vector>

auto hikari::ShapeRectangle::create() -> std::shared_ptr<ShapeRectangle>
{
  return std::shared_ptr<ShapeRectangle>(new ShapeRectangle());
}

hikari::ShapeRectangle::~ShapeRectangle() noexcept
{
}

hikari::ShapeRectangle::ShapeRectangle():Shape()
{
}

hikari::Uuid hikari::ShapeRectangle::getID() const
{
  return ID();
}

auto hikari::ShapeRectangle::createMesh() -> std::shared_ptr<Shape>
{
  //-1+1   +1+1 
  //3_______2   
  // |   ./|    
  // | ./  |    
  // |/____|    
  //0       1   
  //-1-1  +1-1  
  std::vector<hikari::Vec3> vertices = {
  hikari::Vec3{-1.0f,-1.0f,0.0f},
  hikari::Vec3{+1.0f,-1.0f,0.0f},
  hikari::Vec3{+1.0f,+1.0f,0.0f},
  hikari::Vec3{-1.0f,+1.0f,0.0f}
  };
  std::vector<hikari::Vec3> normals  = {
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f},
    hikari::Vec3{0.0f,0.0f,1.0f}
  };
  std::vector<hikari::Vec2> uvs      = {
    hikari::Vec2{ 0.0f, 0.0f},
    hikari::Vec2{+1.0f, 0.0f},
    hikari::Vec2{+1.0f,+1.0f},
    hikari::Vec2{ 0.0f,+1.0f}
  };
  std::vector<hikari::U32>  faces    = { 0,1,2, 2,3,0 };

  auto res = ShapeMesh::create();
  res->setVertexPositions(vertices);
  res->setVertexNormals(normals);
  res->setVertexUVs(uvs);
  res->setFaces(faces);
  return res;
}
