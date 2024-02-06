#pragma once
#include <hikari/core/transform.h>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <memory>
#include <variant>
#include <optional>
#include <vector>
#include <string>
namespace hikari {
  namespace core {
    struct Scene;
    struct Object;
    struct Object    : public std::enable_shared_from_this<Object> {
      static auto create(const std::string& name, const Transform& transform) -> std::shared_ptr<Object> {
        return std::shared_ptr<Object>(new Object(name, transform));
      }
      virtual ~Object() noexcept {}

      auto toString() const->std::string;

      auto getName() const -> std::string   { return m_name; }
      void setName(const std::string& name) { m_name = name; }

      auto getParent() const -> std::shared_ptr<Object> { return m_parent.lock(); }
      void setParent(const std::shared_ptr<Object>& new_parent) {
        auto old_parent = m_parent.lock();
        if (!old_parent) {
          if (!new_parent) { return; }
          m_parent = new_parent;
          new_parent->m_children.push_back(shared_from_this());
          auto new_parent_transform = new_parent->getGlobalTransform();
          m_global_transform = m_local_transform * new_parent_transform;
          updateGlobalTransform(new_parent_transform);
        }
        else {
          auto iter = std::find(std::begin(old_parent->m_children), std::end(old_parent->m_children), shared_from_this());
          if (iter != std::end(old_parent->m_children)) {
            old_parent->m_children.erase(iter);
          }
          if (!new_parent) { m_parent = {}; return; }
          m_parent = new_parent;
          new_parent->m_children.push_back(shared_from_this());
          auto new_parent_transform = new_parent->getGlobalTransform();
          auto old_global_transform = m_global_transform;
          m_global_transform = m_local_transform * new_parent_transform;
          updateGlobalTransform(old_global_transform.inverse() * m_global_transform);
        }
      }

      auto getChildren()   const -> const std::vector<std::shared_ptr<Object>>   & { return m_children; }

      void setGlobalTransform(const Transform& transform) {
        auto rel_transform  = m_global_transform.inverse() * transform;
        m_global_transform  = transform;// global transform
        auto parent         = getParent();
        if (parent) {
          m_local_transform = m_global_transform * parent->getGlobalTransform().inverse() ;
        }
        else {
          m_local_transform = m_global_transform;
        }
        updateGlobalTransform(rel_transform);
      }
      void getGlobalTransform(Transform& transform)const {
        transform = m_global_transform;
      }
      auto getGlobalTransform() const->Transform {
        return m_global_transform;
      }

      auto getGlobalMatrix() const -> glm::mat4 { return m_global_transform.getMat(); }
      bool getGlobalPosition(glm::vec3& position) {
        auto position_ = m_global_transform.getPosition();
        if (position_) { position = *position_; return true; }
        return false;
      }
      bool getGlobalRotation(glm::quat& rotation) {
        auto rotation_ = m_global_transform.getRotation();
        if (rotation_) { rotation = *rotation_; return true; }
        return false;
      }
      bool getGlobalScale(glm::vec3& scale) {
        auto scale_ = m_global_transform.getScale();
        if (scale_) { scale = *scale_; return true; }
        return false;
      }
      auto getGlobalPosition() const->std::optional<glm::vec3> { return m_global_transform.getPosition(); }
      auto getGlobalRotation() const->std::optional<glm::quat> { return m_global_transform.getRotation(); }
      auto getGlobalScale()    const->std::optional<glm::vec3> { return m_global_transform.getScale(); }

      void setLocalTransform(const Transform& transform) {
        m_local_transform         = transform;
        auto parent               = getParent();
        auto old_global_transform = m_global_transform;
        if (parent) {
          m_global_transform      = m_local_transform * parent->getGlobalTransform();
        }
        else {
          m_global_transform      = m_local_transform;
        }
        updateGlobalTransform(old_global_transform.inverse()*m_global_transform);
      }
      void getLocalTransform(Transform& transform)const {
        transform = m_local_transform;
      }
      auto getLocalTransform() const->Transform { return m_local_transform; }

      auto getLocalMatrix  () const -> glm::mat4 { return m_local_transform.getMat(); }
      bool getLocalPosition(glm::vec3& position) {
        auto position_ = m_local_transform.getPosition();
        if (position_) { position = *position_; return true; }
        return false;
      }
      bool getLocalRotation(glm::quat& rotation) {
        auto rotation_ = m_local_transform.getRotation();
        if (rotation_) { rotation = *rotation_; return true; }
        return false;
      }
      bool getLocalScale   (glm::vec3&    scale) {
        auto scale_ = m_local_transform.getScale();
        if (scale_) { scale = *scale_; return true; }
        return false;
      }
      auto getLocalPosition() const->std::optional<glm::vec3> { return m_local_transform.getPosition(); }
      auto getLocalRotation() const->std::optional<glm::quat> { return m_local_transform.getRotation(); }
      auto getLocalScale()    const->std::optional<glm::vec3> { return m_local_transform.getScale   (); }
    protected:
      Object(const std::string& name, const Transform& transform) noexcept
        : m_name{ name }, m_local_transform{ transform }, m_global_transform{transform} {}
      void updateGlobalTransform(const Transform& rel_transform) {
        for (auto& child : m_children) {
          child->m_global_transform = child->m_global_transform * rel_transform;
          child->updateGlobalTransform(rel_transform);
        }
      }
    private:
      std::string                              m_name             = "";
      std::weak_ptr<Scene>                     m_scene            = {};
      std::weak_ptr<Object>                    m_parent           = {};
      std::vector<std::shared_ptr<Object>>     m_children         = {};
      Transform                                m_local_transform  = Transform();
      Transform                                m_global_transform = Transform();
    };
  }
}
