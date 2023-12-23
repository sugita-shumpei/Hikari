#include <hikari/core/scene.h>
#include <hikari/core/node.h>

#include <hikari/shape/rectangle.h>
#include <hikari/shape/triangle.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/cube.h>
#include <hikari/shape/mesh.h>

int main() {
  auto scene    = hikari::Scene::create();
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->getChild(0)->setShape(hikari::ShapeRectangle::create());
  scene->getChild(1)->setShape(hikari::ShapeTriangle::create({ 0.0f,0.0f,0.0f }, { 2.0f,0.0f,0.0f }, { 0.0f,2.0f,0.0f }));
  scene->getChild(2)->setShape(hikari::ShapeSphere::create({0.0f,0.0f,0.0f},1.0f));
  scene->getChild(3)->setShape(hikari::ShapeCube::create());
  scene->getChild(4)->setShape(hikari::ShapeMesh::create());
  auto t0 = scene->getChild(0)->getShape()->convert<hikari::ShapeRectangle>();
  auto t1 = scene->getChild(1)->getShape()->convert<hikari::ShapeTriangle>();
}
