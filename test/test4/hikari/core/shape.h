#pragma once
#include <hikari/core/node.h>
#include <hikari/core/bbox.h>
#include <hikari/core/object.h>
#include <hikari/core/serializer.h>
#include <hikari/core/deserializer.h>
namespace hikari {
  inline namespace core {
    // ShapeObject...形状を管理するためのObject
    // 
    struct ShapeObject : public Object {
      using base_type = Object;
      static inline constexpr const char* TypeString() { return "Shape"; }
      static inline Bool Convertible(const Str& str) noexcept {
        if (base_type::Convertible(str)) { return true; }
        return str == TypeString();
      }
      virtual ~ShapeObject() {}
      virtual auto getBBox() const->BBox3 = 0;
    };
    // Shape...
    //
    template<typename ShapeObjectT, std::enable_if_t<std::is_base_of_v<ShapeObject,ShapeObjectT>,nullptr_t> = nullptr>
    struct ShapeImpl : protected ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, ShapeObjectT> {
      using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, ShapeObjectT>;
      using type = typename impl_type::type;

      ShapeImpl() noexcept : impl_type() {}
      ShapeImpl(nullptr_t) noexcept : impl_type(nullptr) {}
      ShapeImpl(const ShapeImpl&) = default;
      ShapeImpl(const std::shared_ptr<ShapeObjectT>& object) : impl_type(object) {}
      //ShapeImpl& operator=(const Shape&) = default;
      //ShapeImpl(const std::shared_ptr<ShapeObject>& object) : impl_type(object) {}
      //ShapeImpl& operator=(const std::shared_ptr<ShapeObject>& obj)
      //{
      //  setObject(obj);
      //  return *this;
      //}

      HK_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(getBBox, BBox3, {});

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
    };
    struct Shape : protected ShapeImpl<ShapeObject> {
      using impl_type = ShapeImpl<ShapeObject>;
      using type      = ShapeObject;

      Shape() noexcept : impl_type() {}
      Shape(nullptr_t) noexcept : impl_type(nullptr) {}
      Shape(const Shape&) = default;
      Shape& operator=(const Shape&) = default;
      Shape(const std::shared_ptr<ShapeObject>& object) : impl_type(object) {}
      Shape& operator=(const std::shared_ptr<ShapeObject>& obj)
      {
        setObject(obj);
        return *this;
      }
      HK_METHOD_OVERLOAD_COMPARE_OPERATORS(Shape);

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
      using impl_type::getBBox;
    };
    // FilterObject...シーンに対してShapeを追加するためのObject
    // 
    struct ShapeFilterObject   : public NodeComponentObject {
      using base_type = NodeComponentObject;
      static inline constexpr const char* TypeString() { return "ShapeFilter"; }
      static inline Bool Convertible(const Str& str) noexcept {
        if (base_type::Convertible(str)) { return true; }
        return str == TypeString();
      }
      static auto create(const std::shared_ptr<NodeObject>& node) -> std::shared_ptr<ShapeFilterObject> {
        return std::shared_ptr<ShapeFilterObject>(new ShapeFilterObject(node,nullptr));
      }
      template<typename ShapeT, std::enable_if_t<std::is_base_of_v<ShapeObject,typename ShapeT::type>,nullptr_t> = nullptr>
      static auto create(const std::shared_ptr<NodeObject>& node, const ShapeT& shape) -> std::shared_ptr<ShapeFilterObject> {
        return std::shared_ptr<ShapeFilterObject>(new ShapeFilterObject(node, shape.getObject()));
      }
      virtual ~ShapeFilterObject() {}
      Str  getTypeString() const noexcept override;
      Bool isConvertible(const Str& type) const noexcept override;
      auto getPropertyNames() const->std::vector<Str> override;
      void getPropertyBlock(PropertyBlockBase<Object>& pb) const override;
      void setPropertyBlock(const PropertyBlockBase<Object>& pb) override;
      Bool hasProperty(const Str& name) const override;
      Bool getProperty(const Str& name, PropertyBase<Object>& prop) const override;
      Bool setProperty(const Str& name, const PropertyBase<Object>& prop) override;
      auto getNode() const->std::shared_ptr<NodeObject> override;
      void setShape(const std::shared_ptr<ShapeObject>& shape);
      auto getShape() const -> std::shared_ptr<ShapeObject>;
    private:
      ShapeFilterObject(const std::shared_ptr<NodeObject>& node, const std::shared_ptr<ShapeObject>& shape) noexcept
        :NodeComponentObject(), m_node{node}, m_shape{ shape }
      {}
    private:
      std::shared_ptr<ShapeObject> m_shape;
      std::weak_ptr<NodeObject> m_node;
    };
    // Filter...
    //
    struct ShapeFilter : protected NodeComponentImpl<ShapeFilterObject> {
      using impl_type = NodeComponentImpl<ShapeFilterObject>;
      using type = typename impl_type::type;

      ShapeFilter() noexcept : impl_type() {}
      ShapeFilter(nullptr_t) noexcept : impl_type() {}
      ShapeFilter(const ShapeFilter&) = default;
      ShapeFilter& operator=(const ShapeFilter&) = default;
      ShapeFilter(const std::shared_ptr<ShapeFilterObject>& object) : impl_type(object) {}
      ShapeFilter& operator=(const std::shared_ptr<ShapeFilterObject>& obj)
      {
        setObject(obj);
        return *this;
      }
      ~ShapeFilter() noexcept {}

      auto getShape() const -> Shape {
        auto object = getObject();
        if (object) { return Shape(object->getShape()); }
        else { return Shape(); }
      }
      HK_METHOD_OVERLOAD_COMPARE_OPERATORS(ShapeFilter);

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
    // RenderObject...実際に描画対象となるObjectに対して設定する
    // 
    struct ShapeRenderObject   : public NodeComponentObject {
      using base_type = NodeComponentObject;
      static inline constexpr const char* TypeString() { return "ShapeRender"; }
      static inline Bool Convertible(const Str& str) noexcept {
        if (base_type::Convertible(str)) { return true; }
        return str == TypeString();
      }
      static auto create(const std::shared_ptr<NodeObject>& node) -> std::shared_ptr<ShapeRenderObject> {
        return std::shared_ptr<ShapeRenderObject>(new ShapeRenderObject(node));
      }
      virtual ~ShapeRenderObject() {}
      Str  getTypeString() const noexcept override;
      Bool isConvertible(const Str& type) const noexcept override;
      auto getPropertyNames() const->std::vector<Str> override;
      void getPropertyBlock(PropertyBlockBase<Object>& pb) const override;
      void setPropertyBlock(const PropertyBlockBase<Object>& pb) override;
      Bool hasProperty(const Str& name) const override;
      Bool getProperty(const Str& name, PropertyBase<Object>& prop) const override;
      Bool setProperty(const Str& name, const PropertyBase<Object>& prop) override;
      auto getNode() const->std::shared_ptr<NodeObject> override;
    private:
      ShapeRenderObject(const std::shared_ptr<NodeObject>& node) noexcept
        :NodeComponentObject(), m_node{ node }
      {}
    private:
      std::weak_ptr<NodeObject> m_node;
    };
    // Render...
    //
    struct ShapeRender : protected NodeComponentImpl<ShapeRenderObject> {
      using impl_type = NodeComponentImpl<ShapeRenderObject>;
      using type = typename impl_type::type;

      ShapeRender() noexcept : impl_type() {}
      ShapeRender(nullptr_t) noexcept : impl_type() {}
      ShapeRender(const ShapeRender&) = default;
      ShapeRender& operator=(const ShapeRender&) = default;
      ShapeRender(const std::shared_ptr<ShapeRenderObject>& object) : impl_type(object) {}
      ShapeRender& operator=(const std::shared_ptr<ShapeRenderObject>& obj)
      {
        setObject(obj);
        return *this;
      }
      ~ShapeRender() noexcept {}
      HK_METHOD_OVERLOAD_COMPARE_OPERATORS(ShapeRender);

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
    // ShapeFilterSerializer
    //
    struct ShapeFilterSerializer : public ObjectSerializer {
      virtual ~ShapeFilterSerializer() noexcept {}
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    // ShapeFilterDeserializer
    //
    struct ShapeFilterDeserializer : public NodeComponentDeserializer {
      virtual ~ShapeFilterDeserializer() noexcept {}
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<NodeObject>& node, const Json& json) const->std::shared_ptr<NodeComponentObject> override;
    };
    // ShapeRenderSerializer
    //
    struct ShapeRenderSerializer : public ObjectSerializer {
      virtual ~ShapeRenderSerializer() noexcept {}
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
    // ShapeRenderDeserializer
    //
    struct ShapeRenderDeserializer : public NodeComponentDeserializer {
      virtual ~ShapeRenderDeserializer() noexcept {}
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<NodeObject>& node, const Json& json) const->std::shared_ptr<NodeComponentObject> override;
    };
  }
}
