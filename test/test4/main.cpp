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
#include <hikari/render/gfx_window.h>
#include <glad/glad.h>
#include <hikari/platform/common/render/gfx_common_instance.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
using namespace hikari;
void registerObjects(){
  auto& serialize_manager                  = ObjectSerializeManager::getInstance();
  auto& deserialize_manager                = ObjectDeserializeManager::getInstance();
  auto& node_component_deserialize_manager = NodeComponentDeserializeManager::getInstance();
  // Serializer
  serialize_manager.add(std::make_shared<NodeSerializer>());
  serialize_manager.add(std::make_shared<FieldSerializer>());
  serialize_manager.add(std::make_shared<SpectrumRegularSerializer>());
  serialize_manager.add(std::make_shared<SpectrumIrregularSerializer>());
  serialize_manager.add(std::make_shared<SpectrumUniformSerializer>());
  serialize_manager.add(std::make_shared<SpectrumColorSerializer>());
  serialize_manager.add(std::make_shared<ShapeFilterSerializer>());
  serialize_manager.add(std::make_shared<ShapeRenderSerializer>());
  serialize_manager.add(std::make_shared<ShapeMeshSerializer>());
  serialize_manager.add(std::make_shared<ShapeRectSerializer>());
  serialize_manager.add(std::make_shared<ShapeCubeSerializer>());
  serialize_manager.add(std::make_shared<ShapeSphereSerializer>());
  // Deserializer
  deserialize_manager.add(std::make_shared<NodeDeserializer>());
  deserialize_manager.add(std::make_shared<FieldDeserializer>());
  deserialize_manager.add(std::make_shared<SpectrumRegularDeserializer>());
  deserialize_manager.add(std::make_shared<SpectrumIrregularDeserializer>());
  deserialize_manager.add(std::make_shared<SpectrumUniformDeserializer>());
  deserialize_manager.add(std::make_shared<SpectrumColorDeserializer>());
  deserialize_manager.add(std::make_shared<ShapeMeshDeserializer>());
  deserialize_manager.add(std::make_shared<ShapeRectDeserializer>());
  deserialize_manager.add(std::make_shared<ShapeCubeDeserializer>());
  deserialize_manager.add(std::make_shared<ShapeSphereDeserializer>());
  node_component_deserialize_manager.add(std::make_shared<ShapeFilterDeserializer>());
  node_component_deserialize_manager.add(std::make_shared<ShapeRenderDeserializer>());
  return;
}
int main(int argc, const char** argv) {
  // Instanceの作成
  auto  instance       = hikari::platforms::common::render::createGFXInstance(hikari::render::GFXAPI::eOpenGL);
  // WindowManagerの取得
  auto& window_manager = hikari::render::GFXWindowManager::getInstance();
  // Windowの作成
  auto window = instance.createWindow(hikari::render::GFXWindowDesc{ 800,600 });
  window.setTitle("window");
  window.setBorderless(false);
  // Windowの表示
  window.show();
  // MainLoopの実行
  while (true) {
    window_manager.update();
    bool is_window_open = !window.isClosed();
    if (!is_window_open) { window.hide(); window = nullptr; break; }
    if (window) {
      if (window.isFocused()) { std::cout << "window Focused!\n"; }
    }
    auto clip_board = window.getClipboard();
    if (clip_board != "") { std::cout << "clipboard: " << clip_board << std::endl; }

    {
      auto& glfw_manager = hikari::platforms::glfw::WindowManager::getInstance();
      glfw_manager.setCurrentContext(window.getHandle());
      {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
      }
      glfw_manager.swapBuffers(window.getHandle());
    }
  }
  return 0;
}
