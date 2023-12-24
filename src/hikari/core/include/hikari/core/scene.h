#pragma once
#include <hikari/core/data_type.h>
#include <vector>
#include <memory>
namespace hikari {
  struct Camera;
  struct Light;
  struct Shape;
  struct Material;
  struct Node; 
  struct Scene : public std::enable_shared_from_this<Scene>{
    static auto create(const String& name = "") -> std::shared_ptr<Scene>;
    virtual ~Scene();

    void addChild(const std::shared_ptr<Node>& node);
    auto getChildren() -> std::vector<std::shared_ptr<Node>>;
    auto getChildCount() const->U32;
    auto getChild(U32 idx) -> std::shared_ptr<Node>;
    
    void setName(const String& name);
    auto getName() const->String;

    auto getNodesInHierarchy() -> std::vector<std::shared_ptr<Node>>;
    auto getCameras() -> std::vector<std::shared_ptr<Camera>>;
    auto getLights () -> std::vector<std::shared_ptr<Light>> ;
    auto getShapes () -> std::vector<std::shared_ptr<Shape>> ;
    // auto getConfig() ->Config
  private:
    friend struct Node;
    Scene(const String& name);
  private:
    std::shared_ptr<Node> m_root_node;
  };
  using ScenePtr = std::shared_ptr<Scene>;
}
