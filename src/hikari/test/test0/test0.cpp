#include <hikari/core/scene.h>
#include <hikari/core/node.h>
#include <hikari/core/material.h>
#include <hikari/camera/perspective.h>
#include <hikari/light/envmap.h>
#include <hikari/light/area.h>
#include <hikari/shape/sphere.h>
#include <hikari/bsdf/diffuse.h>
#include <hikari/bsdf/conductor.h>
#include <hikari/bsdf/dielectric.h>
#include <hikari/film/hdr.h>
int main(int argc, const char** argv)
{
  auto scene  =   hikari::Scene::create("scene0");

  scene->addChild(hikari::Node ::create("camera"));
  scene->addChild(hikari::Node ::create("envmap"));
  scene->addChild(hikari::Node ::create("backward"));
  scene->addChild(hikari::Node ::create("floor"));
  scene->addChild(hikari::Node ::create("light"));
  scene->addChild(hikari::Node ::create("right"));
  scene->addChild(hikari::Node ::create("left"));

  scene->getChild(0)->setCamera(hikari::CameraPerspective::create());
  scene->getChild(0)->getCamera()->setFilm(hikari::FilmHdr::create());

  scene->getChild(1)->setLight (hikari::LightEnvmap::create());

  scene->getChild(2)->setShape(hikari::ShapeSphere::create());
  scene->getChild(2)->setMaterial(hikari::Material::create());
  scene->getChild(2)->getMaterial()->setBsdf(hikari::BsdfDiffuse::create());

  scene->getChild(3)->setShape(hikari::ShapeSphere::create());
  scene->getChild(3)->setMaterial(hikari::Material::create());
  scene->getChild(3)->getMaterial()->setBsdf(hikari::BsdfDiffuse::create());

  scene->getChild(4)->setShape(hikari::ShapeSphere::create());
  scene->getChild(4)->setLight(hikari::LightArea::create());

  scene->getChild(5)->setShape(hikari::ShapeSphere::create());
  scene->getChild(5)->setMaterial(hikari::Material::create());
  scene->getChild(5)->getMaterial()->setBsdf(hikari::BsdfDiffuse::create());

  scene->getChild(6)->setShape(hikari::ShapeSphere::create());
  scene->getChild(6)->setMaterial(hikari::Material::create());
  scene->getChild(6)->getMaterial()->setBsdf(hikari::BsdfDiffuse::create());

  auto nodes   = scene->getNodesInHierarchy();
  auto cameras = scene->getCameras();
  auto lights  = scene->getLights();
  auto shapes  = scene->getShapes();
  return 0;
}
