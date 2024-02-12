#pragma once
#include <type_traits>
#include <algorithm>
#include <memory>
#include <hikari/core/data_type.h>
#include <hikari/core/object.h>
#include <hikari/core/transform.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari
{
    inline namespace core
    {
        struct NodeObject;
        struct NodeComponentObject : public Object
        {
            using base_type = Object;
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                if (str == TypeString())
                {
                    return true;
                }
                return false;
            }
            static inline constexpr auto TypeString() -> const char * { return "NodeComponent"; }
            virtual ~NodeComponentObject() noexcept;
            virtual auto getName() const -> Str override final;
            virtual auto getNode() const -> std::shared_ptr<NodeObject> = 0;
        };

        template <typename NodeT, typename NodeComponentObjectT, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentObjectT>, nullptr_t> = nullptr>
        struct NodeComponentImplBase : protected ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef, NodeComponentObjectT>
        {
            using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef, NodeComponentObjectT>;
            using type = NodeComponentObjectT;

            NodeComponentImplBase() noexcept : impl_type() {}
            NodeComponentImplBase(nullptr_t) noexcept : impl_type(nullptr) {}
            NodeComponentImplBase(const NodeComponentImplBase &) noexcept = default;
            NodeComponentImplBase(const std::shared_ptr<NodeComponentObjectT> &object) noexcept : impl_type(object) {}
            ~NodeComponentImplBase() noexcept {}

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getNode, NodeT, nullptr);

            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::operator[];
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setObject;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;
        };
        template <typename NodeT, typename NodeComponentObjectT>
        struct NodeComponentBase : protected NodeComponentImplBase<NodeT, NodeComponentObjectT>
        {
            using impl_type = NodeComponentImplBase<NodeT, NodeComponentObjectT>;
            using type = typename impl_type::type;

            NodeComponentBase() noexcept : impl_type() {}
            NodeComponentBase(nullptr_t) noexcept : impl_type(nullptr) {}
            NodeComponentBase(const NodeComponentBase &) noexcept = default;
            NodeComponentBase &operator=(const NodeComponentBase &) noexcept = default;

            NodeComponentBase(const std::shared_ptr<NodeComponentObjectT> &object) noexcept : impl_type(object) {}
            NodeComponentBase &operator=(const std::shared_ptr<NodeComponentObjectT> &obj) noexcept
            {
                setObject(obj);
                return *this;
            }
            ~NodeComponentBase() noexcept {}

            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::operator[];
            using impl_type::getName;
            using impl_type::getNode;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;
        };

        struct Node;
        struct NodeObject : public Object, std::enable_shared_from_this<NodeObject>
        {
            using base_type = Object;
            using wrapper_type = Node;
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                if (str == TypeString())
                {
                    return true;
                }
                return false;
            }
            static inline constexpr auto TypeString() -> const char * { return "Node"; }

            static auto create(const Str &name, const Transform &transform) -> std::shared_ptr<NodeObject>;
            virtual ~NodeObject() noexcept {}

            virtual auto getTypeString() const noexcept -> Str { return TypeString(); }
            virtual auto getPropertyNames() const -> std::vector<Str> override;
            virtual void setPropertyBlock(const PropertyBlock &pb) override;
            virtual void getPropertyBlock(PropertyBlock &pb) const override;
            virtual Bool hasProperty(const Str &name) const override;
            virtual Bool setProperty(const Str &name, const Property &value) override;
            virtual Bool getProperty(const Str &name, Property &value) const override;
            virtual Bool isConvertible(const Str &type_name) const noexcept override { return Convertible(type_name); }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto addComponent() -> std::shared_ptr<NodeComponentType>
            {
                auto object = shared_from_this();
                auto component = NodeComponentType::create(object);
                if (component)
                {
                    m_components.push_back(std::static_pointer_cast<NodeComponentObject>(component));
                }
                return component;
            }
            template <typename NodeComponentType, typename... Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto addComponent(Args &&...args) -> std::shared_ptr<NodeComponentType>
            {
                auto object = shared_from_this();
                auto component = NodeComponentType::create(object, std::forward<Args &&>(args)...);
                if (component)
                {
                    m_components.push_back(std::static_pointer_cast<NodeComponentObject>(component));
                }
                return component;
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            Bool hasComponent() const
            {
                return hasComponent(NodeComponentType::TypeString());
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            void popComponent()
            {
                return popComponent(NodeComponentType::TypeString());
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponent() const -> std::shared_ptr<NodeComponentType>
            {
                return std::static_pointer_cast<NodeComponentType>(getComponent(NodeComponentType::TypeString()));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponentInParent() const -> std::shared_ptr<NodeComponentType>
            {
                return std::static_pointer_cast<NodeComponentType>(getComponentInParent(NodeComponentType::TypeString()));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponentInChildren() const -> std::shared_ptr<NodeComponentType>
            {
                return std::static_pointer_cast<NodeComponentType>(getComponentInChildren(NodeComponentType::TypeString()));
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponents() const -> std::vector<std::shared_ptr<NodeComponentType>>
            {
                auto components = getComponents(NodeComponentType::TypeString());
                auto res = std::vector<std::shared_ptr<NodeComponentType>>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return std::static_pointer_cast<NodeComponentType>(v); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponentsInParent() const -> std::vector<std::shared_ptr<NodeComponentType>>
            {
                auto components = getComponentsInParent(NodeComponentType::TypeString());
                auto res = std::vector<std::shared_ptr<NodeComponentType>>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return std::static_pointer_cast<NodeComponentType>(v); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, NodeComponentType>, nullptr_t> = nullptr>
            auto getComponentsInChildren() const -> std::vector<std::shared_ptr<NodeComponentType>>
            {
                auto components = getComponentsInChildren(NodeComponentType::TypeString());
                auto res = std::vector<std::shared_ptr<NodeComponentType>>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return std::static_pointer_cast<NodeComponentType>(v); });
                return res;
            }

            Bool hasComponent(const Str &componentType) const;
            void popComponent(const Str &componentType);

            auto getComponent(const Str &componentType) const -> std::shared_ptr<NodeComponentObject>;
            auto getComponentInParent(const Str &componentType) const -> std::shared_ptr<NodeComponentObject>;
            auto getComponentInChildren(const Str &componentType) const -> std::shared_ptr<NodeComponentObject>;

            auto getComponents(const Str &componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>>;
            auto getComponentsInParent(const Str &componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>>;
            auto getComponentsInChildren(const Str &componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>>;

            virtual auto getName() const -> Str override;
            void setName(const Str &name);

            auto getParent() const -> std::shared_ptr<NodeObject>;
            void setParent(const std::shared_ptr<NodeObject> &new_parent);

            auto getChildren() const -> const std::vector<std::shared_ptr<NodeObject>> &;
            void setChildren(const std::vector<std::shared_ptr<NodeObject>> &children);
            void popChildren();

            auto getChild(size_t idx) const -> std::shared_ptr<NodeObject>;
            void setChild(size_t idx, const std::shared_ptr<NodeObject> &child);
            void addChild(const std::shared_ptr<NodeObject> &child);
            void popChild(size_t idx);

            auto getChildCount() const -> size_t;
            void setChildCount(size_t count);

            void setGlobalTransform(const Transform &transform);
            void getGlobalTransform(Transform &transform) const;
            auto getGlobalTransform() const -> Transform;

            auto getGlobalMatrix() const -> glm::mat4;
            bool getGlobalPosition(Vec3 &position);
            bool getGlobalRotation(Quat &rotation);
            bool getGlobalScale(Vec3 &scale);
            auto getGlobalPosition() const -> Option<Vec3>;
            auto getGlobalRotation() const -> Option<Quat>;
            auto getGlobalScale() const -> Option<Vec3>;

            void setLocalTransform(const Transform &transform);
            void getLocalTransform(Transform &transform) const;
            auto getLocalTransform() const -> Transform;

            auto getLocalMatrix() const -> glm::mat4;
            bool getLocalPosition(Vec3 &position);
            bool getLocalRotation(Quat &rotation);
            bool getLocalScale(Vec3 &scale);
            auto getLocalPosition() const -> Option<Vec3>;
            auto getLocalRotation() const -> Option<Quat>;
            auto getLocalScale() const -> Option<Vec3>;

        protected:
            NodeObject(const Str &name, const Transform &transform) noexcept
                : m_name{name}, m_local_transform{transform}, m_global_transform{transform} {}
            void updateGlobalTransform(const Transform &new_global_transform)
            {
                auto prv_global_transform = m_global_transform;
                auto rel_transform = prv_global_transform.inverse() * new_global_transform;
                m_global_transform = new_global_transform;
                updateRelGlobalTransform(rel_transform);
            }
            void updateParentGlobalTransform(const Transform &new_parent_transform)
            {
                updateGlobalTransform(m_local_transform * new_parent_transform);
            }
            void updateRelGlobalTransform(const Transform &rel_transform)
            {
                for (auto &child : m_children)
                {
                    child->m_global_transform = child->m_global_transform * rel_transform;
                    child->updateRelGlobalTransform(rel_transform);
                }
            }

        private:
            Str m_name = "";
            std::weak_ptr<NodeObject> m_parent = {};
            std::vector<std::shared_ptr<NodeObject>> m_children = {};
            std::shared_ptr<NodeComponentObject> m_transform = {};
            std::vector<std::shared_ptr<NodeComponentObject>> m_components = {};
            Transform m_local_transform = Transform();
            Transform m_global_transform = Transform();
        };

        struct NodeRef;
        struct NodeRef;
        struct Node : private ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, NodeObject>
        {
            using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, NodeObject>;
            using type = NodeObject;

            Node() noexcept : impl_type() {}
            Node(nullptr_t) noexcept : impl_type(nullptr) {}
            Node(const std::string &name, const Transform &local_transform = {}) noexcept : impl_type(NodeObject::create(name, local_transform)) {}
            Node(const Node &) = default;
            Node &operator=(const Node &) = default;
            Node(const std::shared_ptr<NodeObject> &object) : impl_type(object) {}
            Node &operator=(const std::shared_ptr<NodeObject> &obj)
            {
                setObject(obj);
                return *this;
            }

            auto operator[](size_t idx) const -> Node;
            auto operator[](size_t idx) -> NodeRef;

            using impl_type::operator[];
            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setObject;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto addComponent() -> NodeComponentType
            {
                auto object = getObject();
                if (object)
                {
                    return NodeComponentType(object->addComponent<NodeComponentType::type>());
                }
                return NodeComponentType();
            }
            template <typename NodeComponentType, typename... Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto addComponent(Args &&...args) -> NodeComponentType
            {
                auto object = getObject();
                if (object)
                {
                    return NodeComponentType(object->addComponent<NodeComponentType::type>(std::forward<Args &&>(args)...));
                }
                return NodeComponentType();
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            Bool hasComponent() const
            {
                return hasComponent(NodeComponentType::type::TypeString());
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            void popComponent()
            {
                return popComponent(NodeComponentType::type::TypeString());
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponent() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponent(NodeComponentType::type::TypeString())));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentInParent() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponentInParent(NodeComponentType::type::TypeString())));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentInChildren() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponentInChildren(NodeComponentType::type::TypeString())));
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponents() const -> std::vector<NodeComponentType>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto components = object->getComponents(NodeComponentType::type::TypeString());
                auto res = std::vector<NodeComponentType>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<typename NodeComponentType::type>(v)); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentsInParent() const -> std::vector<NodeComponentType>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto components = object->getComponentsInParent(NodeComponentType::type::TypeString());
                auto res = std::vector<NodeComponentType>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(v)); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentsInChildren() const -> std::vector<NodeComponentType>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto components = object->getComponentsInChildren(NodeComponentType::type::TypeString());
                auto res = std::vector<NodeComponentType>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(v)); });
                return res;
            }

            Bool hasComponent(const Str &componentType) const
            {
                auto object = getObject();
                if (!object)
                {
                    return false;
                }
                return object->hasComponent(componentType);
            }

            auto getComponent(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
                auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponent(componentType);
            }
            auto getComponentInParent(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
                auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponentInParent(componentType);
            }
            auto getComponentInChildren(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
                auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponentInChildren(componentType);
            }

            auto getComponents(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponents(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentBase<Node, NodeComponentObject>(v); });
                return res;
            }
            auto getComponentsInParent(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponentsInParent(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentBase<Node, NodeComponentObject>(v); });
                return res;
            }
            auto getComponentsInChildren(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponentsInChildren(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentBase<Node, NodeComponentObject>(v); });
                return res;
            }

            auto getSize() const -> size_t { return getChildCount(); }
            void setSize(size_t count) { setChildCount(count); }

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setName, Str);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getChildCount, size_t, 0);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setChildCount, size_t);

            auto getChildren() const -> std::vector<Node>;
            void setChildren(const std::vector<Node> &children);
            void popChildren();

            auto getChild(size_t idx) const -> Node;
            void setChild(size_t idx, const Node &child);
            void addChild(const Node &field);
            void popChild(size_t idx);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setGlobalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(getGlobalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getGlobalTransform, Transform, TransformTRSData());
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getGlobalMatrix, Mat4, 1.0f);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalScale, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalScale, Vec3);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setLocalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(getLocalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getLocalTransform, Transform, TransformTRSData());
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getLocalMatrix, Mat4, 1.0f);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalScale, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalScale   , Vec3);
        };
        struct NodeRef : protected ObjectWrapperRefImpl<impl::ObjectWrapperHolderChildObjectRef, NodeObject>
        {
            using impl_type = ObjectWrapperRefImpl<impl::ObjectWrapperHolderChildObjectRef, NodeObject>;
            using type = NodeObject;
            using Ref = NodeRef;

            NodeRef(const NodeRef &) noexcept = delete;
            NodeRef(NodeRef &&) noexcept = delete;
            NodeRef &operator=(const NodeRef &) = delete;
            NodeRef &operator=(NodeRef &&) = delete;

            auto operator[](size_t idx) const -> Node;
            auto operator[](size_t idx) -> Ref;

            using impl_type::operator[];
            using impl_type::operator!;
            using impl_type::operator=;
            using impl_type::operator bool;
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setObject;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto addComponent() -> NodeComponentType
            {
                auto object = getObject();
                if (object)
                {
                    return NodeComponentType(object->addComponent<typename NodeComponentType::type>());
                }
                return NodeComponentType();
            }
            template <typename NodeComponentType, typename... Args, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto addComponent(Args &&...args) -> NodeComponentType
            {
                auto object = getObject();
                if (object)
                {
                    return NodeComponentType(object->addComponent<NodeComponentType::type>(std::forward<Args &&>(args)...));
                }
                return NodeComponentType();
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            Bool hasComponent() const
            {
                return hasComponent(NodeComponentType::type::TypeString());
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            void popComponent()
            {
                return popComponent(NodeComponentType::type::TypeString());
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponent() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponent(NodeComponentType::type::TypeString())));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentInParent() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponentInParent(NodeComponentType::type::TypeString())));
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentInChildren() const -> NodeComponentType
            {
                auto object = getObject();
                if (!object)
                {
                    return NodeComponentType();
                }
                return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(object->getComponentInChildren(NodeComponentType::type::TypeString())));
            }

            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponents() const -> std::vector<NodeComponentType>
            {
                auto components = getComponents(NodeComponentType::type::TypeString());
                auto res = std::vector<NodeComponentType>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(v)); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentsInParent() const -> std::vector<NodeComponentType>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto components = object->getComponentsInParent(NodeComponentType::type::TypeString());
                auto res = std::vector<std::shared_ptr<NodeComponentType>>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(v)); });
                return res;
            }
            template <typename NodeComponentType, std::enable_if_t<std::is_base_of_v<NodeComponentObject, typename NodeComponentType::type>, nullptr_t> = nullptr>
            auto getComponentsInChildren() const -> std::vector<NodeComponentType>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto components = object->getComponentsInChildren(NodeComponentType::type::TypeString());
                auto res = std::vector<std::shared_ptr<NodeComponentType>>();
                res.reserve(components.size());
                std::transform(std::begin(components), std::end(components), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentType(std::static_pointer_cast<NodeComponentType::type>(v)); });
                return res;
            }

            Bool hasComponent(const Str &componentType) const
            {
                auto object = getObject();
                if (!object)
                {
                    return false;
                }
                return object->hasComponent(componentType);
            }

            auto getComponent(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
                auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponent(componentType);
            }
            auto getComponentInParent(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
                auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponentInParent(componentType);
            }
            auto getComponentInChildren(const Str &componentType) const -> NodeComponentBase<Node, NodeComponentObject>
            {
              auto object = getObject();
                if (!object)
                {
                    return nullptr;
                }
                return object->getComponentInChildren(componentType);
            }

            auto getComponents(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
              auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponents(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentBase<Node, NodeComponentObject>(v); });
                return res;
            }
            auto getComponentsInParent(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponentsInParent(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponent(v); });
                return res;
            }
            auto getComponentsInChildren(const Str &componentType) const -> std::vector<NodeComponentBase<Node, NodeComponentObject>>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                auto children = object->getComponentsInChildren(componentType);
                std::vector<NodeComponentBase<Node, NodeComponentObject>> res;
                res.reserve(children.size());
                std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto &v)
                               { return NodeComponentBase<Node, NodeComponentObject>(v); });
                return res;
            }


            auto getSize() const->size_t { return getChildCount(); }
            void setSize(size_t count) { return setChildCount(count); }

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getChildCount, size_t, 0);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setChildCount, size_t);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setName, Str);

            auto getChildren() const -> std::vector<Node>;
            void setChildren(const std::vector<Node> &children);
            void popChildren();

            auto getChild(size_t idx) const -> Node;
            void setChild(size_t idx, const Node &child);
            void addChild(const Node &field);
            void popChild(size_t idx);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setGlobalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(getGlobalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getGlobalTransform, Transform, TransformTRSData());
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getGlobalMatrix, Mat4, 1.0f);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getGlobalScale, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getGlobalScale, Vec3);

            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(setLocalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(getLocalTransform, Transform);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getLocalTransform, Transform, TransformTRSData());
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getLocalMatrix, Mat4, 1.0f);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(getLocalScale, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalPosition, Vec3);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalRotation, Quat);
            HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(getLocalScale, Vec3);
        private:
            friend class Node;
            NodeRef(const std::shared_ptr<NodeObject> &object, size_t idx) : impl_type(impl::ObjectWrapperHolderChildObjectRef(object, idx)) {}
        };
        using  NodeComponent = NodeComponentBase<Node, NodeComponentObject>;

        struct NodeTransformObject : public NodeComponentObject
        {
            using base_type = NodeComponentObject;
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (base_type::Convertible(str))
                {
                    return true;
                }
                if (str == TypeString())
                {
                    return true;
                }
                return false;
            }
            static inline constexpr auto TypeString() -> const char * { return "NodeTransform"; }
            virtual ~NodeTransformObject() noexcept {}
            virtual auto getTypeString() const noexcept -> Str override { return TypeString(); }
            virtual auto getNode() const -> std::shared_ptr<NodeObject> override { return m_node.lock(); }

            Bool isConvertible(const Str &type) const noexcept override
            {
                return Convertible(type);
            }
            auto getPropertyNames() const -> std::vector<Str> override
            {
                return std::vector<Str>();
            }
            void getPropertyBlock(PropertyBlockBase<Object> &pb) const override
            {
            }
            void setPropertyBlock(const PropertyBlockBase<Object> &pb) override
            {
            }
            Bool hasProperty(const Str &name) const override
            {
                return Bool();
            }
            Bool getProperty(const Str &name, PropertyBase<Object> &prop) const override
            {
                return Bool();
            }
            Bool setProperty(const Str &name, const PropertyBase<Object> &prop) override
            {
                return Bool();
            }

            void setGlobalTransform(const Transform &transform);
            void getGlobalTransform(Transform &transform) const;
            auto getGlobalTransform() const -> Transform;

            auto getGlobalMatrix() const -> glm::mat4;
            bool getGlobalPosition(Vec3 &position);
            bool getGlobalRotation(Quat &rotation);
            bool getGlobalScale(Vec3 &scale);
            auto getGlobalPosition() const -> Option<Vec3>;
            auto getGlobalRotation() const -> Option<Quat>;
            auto getGlobalScale() const -> Option<Vec3>;

            void setLocalTransform(const Transform &transform);
            void getLocalTransform(Transform &transform) const;
            auto getLocalTransform() const -> Transform;

            auto getLocalMatrix() const -> glm::mat4;
            bool getLocalPosition(Vec3 &position);
            bool getLocalRotation(Quat &rotation);
            bool getLocalScale(Vec3 &scale);
            auto getLocalPosition() const -> Option<Vec3>;
            auto getLocalRotation() const -> Option<Quat>;
            auto getLocalScale() const -> Option<Vec3>;

        protected:
            friend class NodeObject;
            NodeTransformObject(const std::shared_ptr<NodeObject> &node) : NodeComponentObject(), m_node{node} {}

        private:
            std::weak_ptr<NodeObject> m_node;
        };
        struct NodeTransform : protected ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef, NodeTransformObject>
        {
            using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderWeakRef, NodeTransformObject>;
            using type = typename impl_type::type;

            NodeTransform() noexcept : impl_type() {}
            NodeTransform(nullptr_t) noexcept : impl_type() {}
            NodeTransform(const NodeTransform &) = default;
            NodeTransform &operator=(const NodeTransform &) = default;
            NodeTransform(const std::shared_ptr<NodeTransformObject> &object) : impl_type(object) {}
            NodeTransform &operator=(const std::shared_ptr<NodeTransformObject> &obj)
            {
                setObject(obj);
                return *this;
            }
            ~NodeTransform() noexcept {}

            using impl_type::operator!;
            using impl_type::operator bool;
            using impl_type::operator[];
            using impl_type::getName;
            using impl_type::getObject;
            using impl_type::getPropertyBlock;
            using impl_type::getPropertyNames;
            using impl_type::getValue;
            using impl_type::hasValue;
            using impl_type::isConvertible;
            using impl_type::setPropertyBlock;
            using impl_type::setValue;

            void setGlobalTransform(const Transform &transform);
            void getGlobalTransform(Transform &transform) const;
            auto getGlobalTransform() const -> Transform;

            auto getGlobalMatrix() const -> glm::mat4;
            bool getGlobalPosition(Vec3 &position);
            bool getGlobalRotation(Quat &rotation);
            bool getGlobalScale(Vec3 &scale);
            auto getGlobalPosition() const -> Option<Vec3>;
            auto getGlobalRotation() const -> Option<Quat>;
            auto getGlobalScale() const -> Option<Vec3>;

            void setLocalTransform(const Transform &transform);
            void getLocalTransform(Transform &transform) const;
            auto getLocalTransform() const -> Transform;

            auto getLocalMatrix() const -> glm::mat4;
            bool getLocalPosition(Vec3 &position);
            bool getLocalRotation(Quat &rotation);
            bool getLocalScale(Vec3 &scale);
            auto getLocalPosition() const -> Option<Vec3>;
            auto getLocalRotation() const -> Option<Quat>;
            auto getLocalScale() const -> Option<Vec3>;
        };

        HK_TYPE_2_STRING_DEFINE(Node);
        HK_TYPE_2_STRING_DEFINE(NodeComponent);
        HK_TYPE_2_STRING_DEFINE(NodeTransform);

        struct NodeSerializer : public ObjectSerializer
        {
            virtual ~NodeSerializer() noexcept;
            auto getTypeString() const noexcept -> Str override;
            auto eval(const std::shared_ptr<Object> &object) const -> Json override;
        };
        struct NodeDeserializer : public ObjectDeserializer
        {
            virtual ~NodeDeserializer() noexcept;
            auto getTypeString() const noexcept -> Str override;
            virtual auto eval(const Json &json) const -> std::shared_ptr<Object> override;
        };
    }
}
