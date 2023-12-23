#include <hikari/core/scene.h>
#include <hikari/core/node.h>
#include <hikari/camera/orthographic.h>
#include <hikari/camera/perspective.h>

int main() {
  auto scene = hikari::Scene::create();
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->getChild(0)->setCamera(hikari::CameraOrthographic::create());
  scene->getChild(1)->setCamera(hikari::CameraPerspective::create());
}
