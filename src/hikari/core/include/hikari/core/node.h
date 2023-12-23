#pragma once
#include <vector>
#include <memory>
#include <hikari/core/data_type.h>
#include <hikari/core/transform.h>
namespace hikari {
  struct Scene ;
  struct Camera;
  struct Light;
  struct Shape;
  struct Material;
  struct Node : public std::enable_shared_from_this<Node>{
    static auto create(const String& name = "", Bool isRootOnly = false) -> std::shared_ptr<Node>;
    virtual ~Node() noexcept;
    // Nameの設定
    auto getName() const->String { return m_name; }
    void setName(const String& name) { m_name = name; }
    // 親
    void setParent(const std::shared_ptr<Node>& parent);
    auto getParent() ->  std::shared_ptr<Node>;
    // 子供
    auto getChildren() -> std::vector<std::shared_ptr<Node>>;
    auto getChildCount() const-> U32;
    auto getChild(U32 idx) -> std::shared_ptr<Node>;
    // Scene
    auto getScene() -> std::shared_ptr<Scene>;
    // Component
    void setCamera(const std::shared_ptr<Camera>& camera);
    auto getCamera() ->  std::shared_ptr<Camera>;
    void setLight(const std::shared_ptr<Light>& light);
    auto getLight() -> std::shared_ptr<Light>;
    void setShape(const std::shared_ptr<Shape>& shape);
    auto getShape()  -> std::shared_ptr<Shape>;
    void setMaterial(const std::shared_ptr<Material>& material);
    auto getMaterial() -> std::shared_ptr<Material>;

    auto getNodesInHierarchy()->std::vector<std::shared_ptr<Node>>;
    auto getCameras()->std::vector<std::shared_ptr<Camera>> ;// 子ノードのカメラをすべて取得する
    auto getLights() ->std::vector<std::shared_ptr<Light>>  ;// 子ノードのライトをすべて取得する
    auto getShapes() ->std::vector<std::shared_ptr<Shape>>  ;// 子ノードの形状  をすべて取得する
    // Transformの設定
    void setLocalTransform(const Transform&  transform);
    Bool setLocalPosition(const Vec3& position) {
      auto trs = TransformTRSData();
      if (m_local_transform.getTRS(trs)) {
        trs.position = position;
        setLocalTransform(trs);
        return true;
      }
      return  false;
    }
    Bool setLocalRotation(const Quat& rotation) {
      auto trs = TransformTRSData();
      if (m_local_transform.getTRS(trs)) {
        trs.rotation = rotation;
        setLocalTransform(trs);
        return true;
      }
      return  false;
    }
    Bool setLocalScale(const Vec3& scale) {
      auto trs = TransformTRSData();
      if (m_local_transform.getTRS(trs)) {
        trs.scale = scale;
        setLocalTransform(trs);
        return true;
      }
      return  false;
    }
    auto getLocalPosition() const -> std::optional<Vec3>;
    auto getLocalRotation() const -> std::optional<Quat>;
    auto getLocalScale   () const -> std::optional<Vec3>;
    auto getLocalTransform() const->Transform;
    void setGlobalTransform(const Transform& transform);
    Bool setGlobalPosition(const Vec3& position) {
      auto trs = TransformTRSData();
      auto global_transform = getGlobalTransform();
      if (global_transform.getTRS(trs)) {
        trs.position = position;
        setGlobalTransform(trs);
        return true;
      }
      return  false;
    }
    Bool setGlobalRotation(const Quat& rotation) {
      auto trs = TransformTRSData();
      auto global_transform = getGlobalTransform();
      if (global_transform.getTRS(trs)) {
        trs.rotation = rotation;
        setGlobalTransform(trs);
        return true;
      }
      return  false;
    }
    Bool setGlobalScale(const Vec3& scale) {
      auto trs = TransformTRSData();
      auto global_transform = getGlobalTransform();
      if (global_transform.getTRS(trs)) {
        trs.scale = scale;
        setGlobalTransform(trs);
        return true;
      }
      return  false;
    }
    auto getGlobalPosition() const -> std::optional<Vec3>;
    auto getGlobalRotation() const->std::optional<Quat>;
    auto getGlobalScale   () const -> std::optional<Vec3>;
    auto getGlobalTransform() const->Transform;
    auto getParentTransform() const->Transform;
  private:
    friend class Scene;
    Node(const String& name,Bool isRootOnly );
    void addChild(const std::shared_ptr<Node>& node);
    void popChild(const std::shared_ptr<Node>& node);
    void onAttachScene(const std::shared_ptr<Scene>& scene);
    void onDetachScene();
    void updateTransform();
  private:
    std::weak_ptr<Scene>               m_scene ;
    std::weak_ptr<Node>                m_parent;
    std::vector<std::shared_ptr<Node>> m_children;
    std::shared_ptr<Camera>            m_camera;
    std::shared_ptr<Light>             m_light;
    std::shared_ptr<Shape>             m_shape;
    std::shared_ptr<Material>          m_material;
    String                             m_name;
    Transform                          m_local_transform ;
    Transform                          m_parent_transform;
    bool                               m_is_root_only;
  };
  using NodePtr = std::shared_ptr<Node>;
}
