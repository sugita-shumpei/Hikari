#include <hikari/core/scene.h>
#include <hikari/core/node.h>
#include <hikari/light/constant.h>
#include <hikari/light/envmap.h>
#include <hikari/light/area.h>

int main() {
  auto scene = hikari::Scene::create() ;
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->addChild(hikari::Node::create());
  scene->getChild(0)->setLight(hikari::LightConstant::create());
  scene->getChild(1)->setLight(hikari::LightEnvmap::create());
  scene->getChild(2)->setLight(hikari::LightArea::create());
}
