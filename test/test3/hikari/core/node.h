#pragma once
#include <type_traits>
#include <hikari/core/data_type.h>
#include <hikari/core/object.h>
#include <hikari/core/transform.h>
#include <hikari/core/json.h>
namespace hikari {
  inline namespace core {
    struct NodeObject;
    struct NodeComponentObject : public Object {
      using BaseType = Object;
      static inline Bool Convertible(const Str& str) noexcept {
        if (BaseType::Convertible(str)) { return true; }
        if (str == TypeString()) { return true; }
        return false;
      }
      static inline constexpr auto TypeString() -> const char* { return "NodeComponent"; }
      virtual ~NodeComponentObject() noexcept;
      virtual auto getName() const->Str override;
      auto getNode() const->std::shared_ptr<NodeObject>;
    protected:
      NodeComponentObject(const std::shared_ptr<NodeObject>& node);
    private:
      std::weak_ptr<NodeObject> m_node;
    };
    struct NodeComponent {
      using TypesTuple = PropertyTypes;
    public:
      using ObjectType = NodeComponentObject;
      template<typename T>
      using Traits = in_tuple<T, TypesTuple>;
      using PropertyRef = ObjectPropertyRef;

      NodeComponent() noexcept :m_object{} {}
      NodeComponent(nullptr_t) noexcept :m_object{ } {}
      NodeComponent(const NodeComponent&) = default;
      NodeComponent& operator=(const NodeComponent&) = default;
      NodeComponent(const std::shared_ptr<NodeComponentObject>& object) :m_object{ object } {}
      NodeComponent& operator=(const std::shared_ptr<ObjectType>& obj) { m_object = obj; return *this; }
      ~NodeComponent() noexcept {}

      template<size_t N>
      auto operator[](const char(&name)[N])->PropertyRef    { return operator[](Str(name)); }
      template<size_t N>
      auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
      auto operator[](const Str& name) -> PropertyRef;
      auto operator[](const Str& name) const->Property;

      Bool operator!() const noexcept { return !getObject(); }
      operator Bool () const noexcept { return getObject() != nullptr; }

      void setPropertyBlock(const PropertyBlock& pb) { auto object = getObject(); if (!object) { return; } return object->setPropertyBlock(pb); }
      void getPropertyBlock(PropertyBlock& pb) const { auto object = getObject(); if (!object) { return; } return object->getPropertyBlock(pb); }

      auto getJSONString() const->std::string { auto object = getObject(); if (!object) { return ""; } return object->getJSONString(); }

      auto getName() const->Str { auto object = getObject(); if (!object) { return ""; } return object->getName(); }

      auto getObject() const -> std::shared_ptr<NodeComponentObject> { return m_object.lock(); }
      auto getTypeString() const->Str { auto object = getObject(); if (!object) { return ""; } return object->getTypeString(); }

      Bool setValue(const Str& name, const Property& prop) { auto object = getObject(); if (!object) { return false; } return object->setProperty(name, prop); }
      Bool getValue(const Str& name, Property& prop) const { auto object = getObject(); if (!object) { return false; } return object->getProperty(name, prop); }
      auto getValue(const Str& name) const -> Property { auto object = getObject(); if (!object) { return Property(); }  Property prop; object->getProperty(name, prop); return prop; }
      Bool hasValue(const Str& name) const { auto object = getObject(); if (!object) { return false; } return object->hasProperty(name); }

      template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
      void setValue(const Str& name, T value) noexcept { return setValue(name, Property(value)); }
      template <size_t N>
      void setValue(const Str& name, const char(&value)[N]) noexcept { setValue(name, Str(value)); }

      Bool isConvertible(const Str& type_name) const {
        auto object = getObject(); if (!object) { return false; }
        return object->isConvertible(type_name);
      }
    private:
      std::weak_ptr<NodeComponentObject> m_object;
    };
    struct NodeTransformObject : public NodeComponentObject {};
    struct NodeTransform {};

    struct NodeObject : public Object, std::enable_shared_from_this<NodeObject> {
      using BaseType = Object;
      static inline Bool Convertible(const Str& str) noexcept {
        if (BaseType::Convertible(str)) { return true; }
        if (str == TypeString()) { return true; }
        return false;
      }
      static inline constexpr auto TypeString() -> const char* { return "Node"; }

      static auto create(const Str& name, const Transform& transform) -> std::shared_ptr<NodeObject> {
        return std::shared_ptr<NodeObject>(new NodeObject(name, transform));
      }
      virtual ~NodeObject() noexcept {}

      virtual auto getTypeString() const -> Str { return TypeString(); }
      virtual auto getJSONString() const -> Str override;
      virtual auto getPropertyNames() const -> std::vector<Str> override;
      virtual void setPropertyBlock(const PropertyBlock& pb) override;
      virtual void getPropertyBlock(PropertyBlock& pb) const override;
      virtual Bool hasProperty(const Str& name) const override;
      virtual Bool setProperty(const Str& name, const Property& value) override;
      virtual Bool getProperty(const Str& name, Property& value) const override;
      virtual Bool isConvertible(const Str& type_name) const override { return Convertible(type_name); }

      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto addComponent()->std::shared_ptr<NodeComponentObjectType> {
        auto object = shared_from_this();
        auto component = NodeComponentObjectType::create(object);
        if (component) { m_components.push_back(std::static_pointer_cast<NodeComponentObject>(component)); }
        return component;
      }
      template<typename NodeComponentObjectType, typename ...Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto addComponent(Args&&... args) -> std::shared_ptr<NodeComponentObjectType> {
        auto object = shared_from_this();
        auto component = NodeComponentObjectType::create(object, std::forward<Args&&>(args)...);
        if (component) { m_components.push_back(std::static_pointer_cast<NodeComponentObject>(component)); }
        return component;
      }

      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      Bool hasComponent() const {
        return hasComponent(NodeComponentObjectType::TypeString());
      }
      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      void popComponent() {
        return popComponent(NodeComponentObjectType::TypeString());
      }

      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponent() const ->std::shared_ptr<NodeComponentObjectType> {
        return std::static_pointer_cast<NodeComponentObjectType>(getComponent(NodeComponentObjectType::TypeString()));
      }
      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponentInParent() const ->std::shared_ptr<NodeComponentObjectType> {
        return std::static_pointer_cast<NodeComponentObjectType>(getComponentInParent(NodeComponentObjectType::TypeString()));
      }
      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponentInChildren() const ->std::shared_ptr<NodeComponentObjectType> {
        return std::static_pointer_cast<NodeComponentObjectType>(getComponentInChildren(NodeComponentObjectType::TypeString()));
      }

      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponents() const -> std::vector<std::shared_ptr<NodeComponentObjectType>> {
        auto components = getComponents(NodeComponentObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentObjectType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return std::static_pointer_cast<NodeComponentObjectType>(v); });
        return res;
      }
      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponentsInParent() const ->std::vector<std::shared_ptr<NodeComponentObjectType>> {
        auto components = getComponentsInParent(NodeComponentObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentObjectType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return std::static_pointer_cast<NodeComponentObjectType>(v); });
        return res;
      }
      template<typename NodeComponentObjectType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectType>, nullptr_t> = nullptr>
      auto getComponentsInChildren() const ->std::vector<std::shared_ptr<NodeComponentObjectType>> {
        auto components = getComponentsInChildren(NodeComponentObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentObjectType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return std::static_pointer_cast<NodeComponentObjectType>(v); });
        return res;
      }

      Bool hasComponent(const Str& componentType) const;
      void popComponent(const Str& componentType);

      auto getComponent(const Str& componentType) const -> std::shared_ptr<NodeComponentObject>;
      auto getComponentInParent(const Str& componentType) const-> std::shared_ptr<NodeComponentObject>;
      auto getComponentInChildren(const Str& componentType) const-> std::shared_ptr<NodeComponentObject>;

      auto getComponents(const Str& componentType) const->std::vector<std::shared_ptr<NodeComponentObject>>;
      auto getComponentsInParent(const Str& componentType) const->std::vector<std::shared_ptr<NodeComponentObject>>;
      auto getComponentsInChildren(const Str& componentType) const->std::vector<std::shared_ptr<NodeComponentObject>>;

      virtual auto getName() const -> Str override;
      void setName(const Str& name);

      auto getParent() const -> std::shared_ptr<NodeObject>;
      void setParent(const std::shared_ptr<NodeObject>& new_parent);

      auto getChildren()   const -> const std::vector<std::shared_ptr<NodeObject>>&;
      void setChildren(const  std::vector<std::shared_ptr<NodeObject>>& children);
      void popChildren();

      auto getChild(size_t idx) const->std::shared_ptr<NodeObject>;
      void setChild(size_t idx, const std::shared_ptr<NodeObject>& child);
      void addChild(const std::shared_ptr<NodeObject>& child);
      void popChild(size_t idx);

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      void setGlobalTransform(const Transform& transform);
      void getGlobalTransform(Transform& transform)const;
      auto getGlobalTransform() const->Transform;

      auto getGlobalMatrix() const -> glm::mat4;
      bool getGlobalPosition(Vec3& position);
      bool getGlobalRotation(Quat& rotation);
      bool getGlobalScale(Vec3& scale);
      auto getGlobalPosition() const->Option<Vec3>;
      auto getGlobalRotation() const->Option<Quat>;
      auto getGlobalScale()    const->Option<Vec3>;

      void setLocalTransform(const Transform& transform);
      void getLocalTransform(Transform& transform)const;
      auto getLocalTransform() const->Transform;

      auto getLocalMatrix() const -> glm::mat4;
      bool getLocalPosition(Vec3& position);
      bool getLocalRotation(Quat& rotation);
      bool getLocalScale(Vec3& scale);
      auto getLocalPosition() const->Option<Vec3>;
      auto getLocalRotation() const->Option<Quat>;
      auto getLocalScale()    const->Option<Vec3>;
    protected:
      NodeObject(const Str& name, const Transform& transform) noexcept
        : m_name{ name }, m_local_transform{ transform }, m_global_transform{ transform } {}
      void updateGlobalTransform(const Transform& new_global_transform) {
        auto prv_global_transform = m_global_transform;
        auto rel_transform        = prv_global_transform.inverse() * new_global_transform;
        m_global_transform        = new_global_transform;
        updateRelGlobalTransform(rel_transform);
      }
      void updateParentGlobalTransform(const Transform& new_parent_transform) {
        updateGlobalTransform(m_local_transform * new_parent_transform);
      }
      void updateRelGlobalTransform(const Transform& rel_transform) {
        for (auto& child : m_children) {
          child->m_global_transform = child->m_global_transform * rel_transform;
          child->updateRelGlobalTransform(rel_transform);
        }
      }
    private:
      Str m_name = "";
      //std::weak_ptr<Scene>                     m_scene = {};
      std::weak_ptr<NodeObject> m_parent = {};
      std::vector<std::shared_ptr<NodeObject>> m_children = {};
      std::vector<std::shared_ptr<NodeComponentObject>> m_components = {};
      Transform m_local_transform = Transform();
      Transform m_global_transform = Transform();
    };
    struct NodeRef;
    struct NodeRef;
    struct Node {
      using ObjectType  = NodeObject;
      using Ref         = NodeRef;
      using PropertyRef = ObjectPropertyRef;

      Node() noexcept :m_object{} {}
      Node(nullptr_t) noexcept :m_object{ nullptr } {}
      Node(const std::string& name, const Transform& local_transform = {}) noexcept : m_object{ NodeObject::create(name,local_transform) } {}
      Node(const Node&) = default;
      Node& operator=(const Node&) = default;
      Node(const std::shared_ptr<NodeObject>& object) :m_object{ object } {}
      Node& operator=(const std::shared_ptr<NodeObject>& obj) { m_object = obj; return *this; }

      Bool operator!() const noexcept { return !m_object; }
      operator Bool () const noexcept { return m_object != nullptr; }

      template<size_t N>
      auto operator[](const char(&name)[N])->PropertyRef { return operator[](Str(name)); }
      template<size_t N>
      auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
      auto operator[](const Str& name)->PropertyRef;
      auto operator[](const Str& name) const->Property;
      auto operator[](size_t idx) const->Node;
      auto operator[](size_t idx)->NodeRef;


      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto addComponent() -> NodeComponentType {
        auto object = getObject();
        if (object) {
          return NodeComponentType(object->addComponent<NodeComponentType::ObjectType>());
        }
        return NodeComponentType();
      }
      template<typename NodeComponentType, typename ...Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto addComponent(Args&&... args) -> NodeComponentType {
        auto object = getObject();
        if (object) {
          return NodeComponentType(object->addComponent<NodeComponentType::ObjectType>(std::forward<Args&&>(args)...));
        }
        return NodeComponentType();
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      Bool hasComponent() const {
        return hasComponent(NodeComponentType::ObjectType::TypeString());
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      void popComponent() {
        return popComponent(NodeComponentType::ObjectType::TypeString());
      }

      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponent() const -> NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponent(NodeComponentType::ObjectType::TypeString())));
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentInParent() const -> NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponentInParent(NodeComponentType::ObjectType::TypeString())));
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentInChildren() const ->NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponentInChildren(NodeComponentType::ObjectType::TypeString())));
      }

      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponents() const -> std::vector<NodeComponentType> {
        auto components = getComponents(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<NodeComponentType>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentsInParent() const ->std::vector<NodeComponentType> {
        auto object = getObject();
        if (!object) { return {}; }
        auto components = object->getComponentsInParent(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentsInChildren() const ->std::vector<NodeComponentType> {
        auto object = getObject();
        if (!object) { return {}; }
        auto components = object->getComponentsInChildren(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }

      Bool hasComponent(const Str& componentType) const {
        auto object = getObject();
        if (!object) { return false; }
        return object->hasComponent(componentType);
      }

      auto getComponent (const Str& componentType) const -> NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponent(componentType);
      }
      auto getComponentInParent(const Str& componentType) const->NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponentInParent(componentType);
      }
      auto getComponentInChildren(const Str& componentType) const->NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponentInChildren(componentType);
      }

      auto getComponents(const Str& componentType) const -> std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponents(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }
      auto getComponentsInParent(const Str& componentType) const->std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponentsInParent(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }
      auto getComponentsInChildren(const Str& componentType) const->std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponentsInChildren(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }

      void setPropertyBlock(const PropertyBlock& pb) ;
      void getPropertyBlock(PropertyBlock& pb) const ;

      auto getJSONString() const -> std::string;

      auto getSize() const->size_t;
      void setSize(size_t count);

      auto getName() const->Str;
      void setName(const Str& name);

      auto getObject() const->std::shared_ptr<ObjectType>;

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<Node>;
      void setChildren(const  std::vector<Node>& children);
      void popChildren();

      auto getChild(size_t idx) const->Node;
      void setChild(size_t idx, const Node& child);
      void addChild(const Node& field);
      void popChild(size_t idx);

      void setGlobalTransform(const Transform& transform);
      void getGlobalTransform(Transform& transform)const;
      auto getGlobalTransform() const->Transform;

      auto getGlobalMatrix() const->glm::mat4;
      bool getGlobalPosition(Vec3& position);
      bool getGlobalRotation(Quat& rotation);
      bool getGlobalScale(Vec3& scale);
      auto getGlobalPosition() const->Option<Vec3>;
      auto getGlobalRotation() const->Option<Quat>;
      auto getGlobalScale()    const->Option<Vec3>;

      void setLocalTransform(const Transform& transform);
      void getLocalTransform(Transform& transform)const;
      auto getLocalTransform() const->Transform;

      auto getLocalMatrix() const->glm::mat4;
      bool getLocalPosition(Vec3& position);
      bool getLocalRotation(Quat& rotation);
      bool getLocalScale(Vec3& scale);
      auto getLocalPosition() const->Option<Vec3>;
      auto getLocalRotation() const->Option<Quat>;
      auto getLocalScale()    const->Option<Vec3>;
    private:
      std::shared_ptr<ObjectType> m_object;
    };
    struct NodeRef {
      using ObjectType  = NodeObject;
      using Ref         = NodeRef;
      using PropertyRef = ObjectPropertyRef;

      NodeRef(const NodeRef&) noexcept = delete;
      NodeRef(NodeRef&&) noexcept = delete;
      NodeRef& operator=(const NodeRef&) = delete;
      NodeRef& operator=(NodeRef&&) = delete;

      void operator=(const Node& node) noexcept;
      operator Node() const { return Node(getObject()); }

      template<size_t N>
      auto operator[](const char(&name)[N])->PropertyRef { return operator[](Str(name)); }
      template<size_t N>
      auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
      auto operator[](const Str& name)->PropertyRef;
      auto operator[](const Str& name) const->Property;
      auto operator[](size_t idx) const->Node;
      auto operator[](size_t idx)->Ref;


      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto addComponent() -> NodeComponentType {
        auto object = getObject();
        if (object) {
          return NodeComponentType(object->addComponent<NodeComponentType::ObjectType>());
        }
        return NodeComponentType();
      }
      template<typename NodeComponentType, typename ...Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto addComponent(Args&&... args) -> NodeComponentType {
        auto object = getObject();
        if (object) {
          return NodeComponentType(object->addComponent<NodeComponentType::ObjectType>(std::forward<Args&&>(args)...));
        }
        return NodeComponentType();
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      Bool hasComponent() const {
        return hasComponent(NodeComponentType::ObjectType::TypeString());
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      void popComponent() {
        return popComponent(NodeComponentType::ObjectType::TypeString());
      }

      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponent() const -> NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponent(NodeComponentType::ObjectType::TypeString())));
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentInParent() const -> NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponentInParent(NodeComponentType::ObjectType::TypeString())));
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentInChildren() const ->NodeComponentType {
        auto object = getObject();
        if (!object) { return NodeComponentType(); }
        return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(object->getComponentInChildren(NodeComponentType::ObjectType::TypeString())));
      }

      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponents() const -> std::vector<NodeComponentType> {
        auto components = getComponents(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<NodeComponentType>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentsInParent() const ->std::vector<NodeComponentType> {
        auto object = getObject();
        if (!object) { return {}; }
        auto components = object->getComponentsInParent(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }
      template<typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::ObjectType>, nullptr_t> = nullptr>
      auto getComponentsInChildren() const ->std::vector<NodeComponentType> {
        auto object = getObject();
        if (!object) { return {}; }
        auto components = object->getComponentsInChildren(NodeComponentType::ObjectType::TypeString());
        auto res = std::vector<std::shared_ptr<NodeComponentType>>();
        res.reserve(components.size());
        std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto& v) { return NodeComponentType(std::static_pointer_cast<NodeComponentType::ObjectType>(v)); });
        return res;
      }

      Bool hasComponent(const Str& componentType) const {
        auto object = getObject();
        if (!object) { return false; }
        return object->hasComponent(componentType);
      }

      auto getComponent(const Str& componentType) const -> NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponent(componentType);
      }
      auto getComponentInParent(const Str& componentType) const->NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponentInParent(componentType);
      }
      auto getComponentInChildren(const Str& componentType) const->NodeComponent {
        auto object = getObject();
        if (!object) { return nullptr; }
        return object->getComponentInChildren(componentType);
      }

      auto getComponents(const Str& componentType) const -> std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponents(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }
      auto getComponentsInParent(const Str& componentType) const->std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponentsInParent(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }
      auto getComponentsInChildren(const Str& componentType) const->std::vector<NodeComponent> {
        auto object = getObject();
        if (!object) { return {}; }
        auto children = object->getComponentsInChildren(componentType);
        std::vector<NodeComponent> res;
        res.reserve(children.size());
        std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& v) { return NodeComponent(v); });
        return res;
      }

      void setPropertyBlock(const PropertyBlock& pb);
      void getPropertyBlock(PropertyBlock& pb) const;

      auto getSize() const->size_t;
      void setSize(size_t     count);

      auto getName() const->Str;
      void setName(const Str& name);

      auto getObject() const->std::shared_ptr<ObjectType>;

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<Node>;
      void setChildren(const  std::vector<Node>& children);
      void popChildren();

      auto getChild(size_t idx) const->Node;
      void setChild(size_t idx, const Node& child);
      void addChild(const Node& field);
      void popChild(size_t idx);

      void setGlobalTransform(const Transform& transform);
      void getGlobalTransform(Transform& transform)const;
      auto getGlobalTransform() const->Transform;

      auto getGlobalMatrix() const->glm::mat4;
      bool getGlobalPosition(Vec3& position);
      bool getGlobalRotation(Quat& rotation);
      bool getGlobalScale(Vec3& scale);
      auto getGlobalPosition() const->Option<Vec3>;
      auto getGlobalRotation() const->Option<Quat>;
      auto getGlobalScale()    const->Option<Vec3>;

      void setLocalTransform(const Transform& transform);
      void getLocalTransform(Transform& transform)const;
      auto getLocalTransform() const->Transform;

      auto getLocalMatrix() const->glm::mat4;
      bool getLocalPosition(Vec3& position);
      bool getLocalRotation(Quat& rotation);
      bool getLocalScale(Vec3& scale);
      auto getLocalPosition() const->Option<Vec3>;
      auto getLocalRotation() const->Option<Quat>;
      auto getLocalScale()    const->Option<Vec3>;
    private:
      friend struct Node;
      NodeRef(const std::shared_ptr<ObjectType>& object, size_t idx) :m_object{ object }, m_idx{ idx } {}
      std::weak_ptr<ObjectType> m_object;
      size_t m_idx;
    };
    

    auto convertStringToNode(const Str& v) -> Node;

    template<>
    struct ConvertFromJSONStringTraits<Node> :std::true_type {
      static auto eval(const Str& str) -> Option<Node> {
        auto node =  convertStringToNode(str);
        if (!node) { return std::nullopt; }
        return node;
      }
    };

    auto convertJSONToNode(const Json& json)      -> Node;
    auto convertNodeToJSON(const Node& node)      -> Json;

    auto convertToJSONString(const core::Node& v) -> Str;
    auto convertToString(const core::Node& v)     -> Str;
  }
}
