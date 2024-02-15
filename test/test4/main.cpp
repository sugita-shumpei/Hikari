#include <hikari/core/node.h>
#include <hikari/core/field.h>
#include <hikari/core/shape.h>
#include <hikari/core/spectrum.h>
#include <hikari/spectrum/regular.h>
#include <hikari/spectrum/irregular.h>
#include <hikari/spectrum/uniform.h>
#include <hikari/spectrum/blackbody.h>
#include <hikari/spectrum/color.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/rect.h>
#include <hikari/shape/cube.h>
#include <hikari/shape/sphere.h>
using namespace hikari;
void registerObjects(){
  // Serializer
  ObjectSerializeManager::getInstance().add(std::make_shared<NodeSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<FieldSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumRegularSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumIrregularSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumUniformSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumColorSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeFilterSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeRenderSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeMeshSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeRectSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeCubeSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<ShapeSphereSerializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<NodeDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<FieldDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumRegularDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumIrregularDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumUniformDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumColorDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<ShapeMeshDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<ShapeRectDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<ShapeCubeDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<ShapeSphereDeserializer>());
  NodeComponentDeserializeManager::getInstance().add(std::make_shared<ShapeFilterDeserializer>());
  NodeComponentDeserializeManager::getInstance().add(std::make_shared<ShapeRenderDeserializer>());
  return;
}
int main() {
  registerObjects();
  Node node("");
  node.setChildCount(4);
  node[0].addComponent<ShapeFilter>(ShapeMesh("mesh"));
  node[1].addComponent<ShapeFilter>(ShapeRect("rect"));
  node[2].addComponent<ShapeFilter>(ShapeCube("cube"));
  node[3].addComponent<ShapeFilter>(ShapeSphere("sphere", Vec3(0.5f), 2.0f));
  auto mesh         = ObjectWrapperUtils::convert<ShapeMesh>(node[0].getComponent<ShapeFilter>().getShape());
  mesh["positions"] = Array<Vec3>({ Vec3(0.0f,0.0f,0.0f)     , Vec3(1.0f,0.0f,0.0f)       , Vec3(1.0f,1.0f,0.0f) , Vec3(0.0f,1.0f,0.0f) });
  mesh["normals"]   = Array<Vec3>({ Vec3(0.0f,0.0f,1.0f)     , Vec3(0.0f,0.0f,1.0f)       , Vec3(0.0f,0.0f,1.0f) , Vec3(0.0f,0.0f,1.0f) });
  mesh["tangents"]  = Array<Vec4>({ Vec4(0.0f,1.0f,0.0f,1.0f), Vec4(0.0f,1.0f , 0.0f,1.0f), Vec4(0.0f,1.0f,0.0f  , 1.0f), Vec4(0.0f,1.0f,0.0f,1.0f) });
  mesh["uv"]        = Array<Vec2>({ Vec2(0.0f,0.0f)          , Vec2(1.0f,0.0f)            , Vec2(1.0f,1.0f)      , Vec2(0.0f,1.0f) });
  mesh["colors"]    = Array<Vec4>({ Vec4(1), Vec4(1), Vec4(1), Vec4(1) });
  mesh["indices"]   = Array<U16>{ 0,1,2,2,3,0 };
  mesh["index_format"] = "U32";
  std::cout << serialize(node).dump() << std::endl;
  std::cout << serialize(deserialize<ObjectWrapper>(serialize(node))).dump() << std::endl;
  return 0;
}
