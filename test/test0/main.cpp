#include <fmt/printf.h>
#include <hikari/core/object.h>
#include <hikari/core/json.h>
#include <hikari/assets/mitsuba/xml/scene.h>
#include <hikari/assets/mitsuba/shape/mesh_data.h>
#include <hikari/assets/mitsuba/spectrum/serialized_data.h>
#include <test0_config.h>
int main(){
  auto object1 = hikari::core::Object::create("1", hikari::TransformTRSData{ glm::vec3(1.0f,0.0f,0.0f) });
  auto object2 = hikari::core::Object::create("2", hikari::TransformTRSData{ glm::vec3(1.0f,0.0f,0.0f) });
  auto object3 = hikari::core::Object::create("3", hikari::TransformTRSData{ glm::vec3(1.0f,0.0f,0.0f) });
  object2->setParent(object1);
  object3->setParent(object2);
   // xml scene
  auto scene_importer      = hikari::assets::mitsuba::XMLScene::Importer::create(HK_DATA_ROOT"/mitsuba-scene/car/scene.xml");
  auto scene_data          = scene_importer->loadScene();
  fmt::print("{}"          , scene_data->toString());
  {// serialized mesh
    auto mesh_importer     = hikari::assets::mitsuba::shape::MeshImporter::create(HK_DATA_ROOT"/mitsuba-data/scenes/matpreview/matpreview.serialized");
    auto mesh_data         = mesh_importer->load();
  }
  {// ply mesh
    auto mesh_importer     = hikari::assets::mitsuba::shape::MeshImporter::create(HK_DATA_ROOT"/mitsuba-scene/lego/meshes/Brick_01_02_001.ply");
    auto mesh_data         = mesh_importer->load();
  }
  {// obj mesh 
    auto mesh_importer     = hikari::assets::mitsuba::shape::MeshImporter::create(HK_DATA_ROOT"/mitsuba-scene/car/models/Mesh000.obj");
    auto mesh_data         = mesh_importer->load();
  }
  {// spectrum
    auto spectrum_importer = hikari::assets::mitsuba::spectrum::SerializedImporter::create(HK_DATA_ROOT"/mitsuba-data/ior/a-C.eta.spd");
    auto spectrum_data     = spectrum_importer->load();
  }
  return 0;
}
