#include <hikari/core/data_type.h>
#include <hikari/core/transform.h>
#include <hikari/core/scene.h>
#include <hikari/core/node.h>
#include <hikari/core/utils.h>
#include <hikari/core/material.h>
#include <glm/gtc/epsilon.hpp>

using namespace std::literals::string_literals;
int main() {
  assert((hikari::splitString("1.2.3"s, '.') == std::vector<hikari::String>{"1"s, "2"s, "3"s}));
  assert((hikari::splitString("1 2 3"s, ' ') == std::vector<hikari::String>{"1"s, "2"s, "3"s}));
  auto scene =    hikari::Scene::create("cornelbox");
  scene->addChild(hikari::Node::create("floor"));
  scene->addChild(hikari::Node::create("red"));
  auto child0 = scene->getChild(0);
  auto child1 = scene->getChild(1);
  child0->setMaterial(hikari::Material::create());
  child1->setMaterial(hikari::Material::create());
  return 0;
}
