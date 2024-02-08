#pragma once
#include <hikari/core/object.h>
#include <hikari/core/serializer.h>
namespace hikari {
  inline namespace core {
    struct Field;
    // Field:
    //
    struct FieldObject : public Object {
      using wrapper_type = Field;
      using base_type    = Object;
      static inline Bool Convertible(const Str& str) noexcept {
        if (base_type::Convertible(str)) { return true; }
        if (str == TypeString()) { return true; }
        return false;
      }
      static inline constexpr auto TypeString() -> const char* { return "Field"; }

      static auto create(Str name = "") -> std::shared_ptr<FieldObject>;
      virtual ~FieldObject() noexcept;

      virtual auto getTypeString() const noexcept -> Str override { return TypeString(); }

      virtual auto getName() const->Str override;
      void setName(const Str& name);

      virtual auto getPropertyNames() const->std::vector<Str> override;
      virtual void setPropertyBlock(const PropertyBlock& pb) override;
      virtual void getPropertyBlock(PropertyBlock& pb) const override;
      virtual Bool hasProperty(const Str& name) const override;
      virtual Bool setProperty(const Str& name, const Property& value) override;
      virtual Bool getProperty(const Str& name, Property& value) const override;
      virtual Bool isConvertible(const Str& type_name) const noexcept override
      {
        return Convertible(type_name);
      }

      auto clone() const ->std::shared_ptr<FieldObject>;

      Bool getPropertyTypeIndex(const Str& name, size_t& type_index) const;

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<std::shared_ptr<FieldObject>>;
      void setChildren(const std::vector<std::shared_ptr<FieldObject>>& children);
      void popChildren();

      auto getChild(size_t idx) const->std::shared_ptr<FieldObject>;
      void setChild(size_t idx, std::shared_ptr<FieldObject> child);
      void addChild(std::shared_ptr<FieldObject> child);
      void popChild(size_t idx);
    private:
      FieldObject(const Str name = "") :Object(), m_name{ name } {}
    private:
      std::string m_name = "";
      std::vector<std::shared_ptr<FieldObject>>  m_children = {};
      PropertyBlock m_property_block = {};
    };
    // Field:
    // FieldObjectをラップし, 階層構造として利用できるようにしたもの
    // 
    struct FieldRef;
    struct Field : protected ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, FieldObject>{
      using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, FieldObject>;

      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      Field() noexcept : impl_type() {}
      Field(const std::string& name) noexcept : impl_type(type::create(name)) {}
      Field(nullptr_t) noexcept : impl_type(nullptr) {}
      Field(const std::shared_ptr<type>& object) noexcept : impl_type(object) {}
      Field(const Field& opb) noexcept : impl_type(opb.getObject()) {}
      Field(Field&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      Field& operator=(const Field& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      Field& operator=(Field&& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
          opb.setObject({});
        }
        return *this;
      }
      Field& operator=(const std::shared_ptr<type>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      Field(const ObjectWrapperLike& wrapper) noexcept : impl_type(wrapper.getObject()) {}
      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      Field& operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      auto operator[](size_t idx) const->Field;
      auto operator[](size_t idx)->FieldRef;

      auto clone() const -> Field;

      void setName(const Str& name);

      auto getSize() const->size_t;
      void setSize(size_t count);

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<Field>;
      void setChildren(const  std::vector<Field>& children);
      void popChildren();

      auto getChild(size_t idx) const->Field;
      void setChild(size_t idx, const Field& child);
      void addChild(const Field& field);
      void popChild(size_t idx);

      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::operator[];
      using impl_type::isConvertible;
      using impl_type::getName;
      using impl_type::getKeys;
      using impl_type::getObject;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getValue;
      using impl_type::hasValue;
      using impl_type::setValue;
    };
    // FieldRef:
    // FieldObjectの弱参照
    // 
    struct FieldRef : private ObjectWrapperRefImpl<impl::ObjectWrapperHolderChildObjectRef,FieldObject> {
      using impl_type         = ObjectWrapperRefImpl<impl::ObjectWrapperHolderChildObjectRef, FieldObject>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type     = typename impl_type::property_type;
      using type              = typename impl_type::type;
      using wrapper_type      = typename impl_type::wrapper_type;

      FieldRef(const FieldRef&) noexcept   = delete;
      FieldRef(FieldRef&&) noexcept        = delete;
      FieldRef& operator=(const FieldRef&) = delete;
      FieldRef& operator=(FieldRef&&)      = delete;

      auto operator[](size_t idx) const->Field;
      auto operator[](size_t idx)-> FieldRef;

      using impl_type::operator=;
      using impl_type::operator[];
      using impl_type::operator wrapper_type;
      using impl_type::operator bool;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getKeys;
      using impl_type::getName;
      using impl_type::getValue;
      using impl_type::setValue;

      auto getSize() const->size_t;
      void setSize(size_t count);

      void setName(const Str& name);

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<Field>;
      void setChildren(const  std::vector<Field>& children);
      void popChildren();

      auto getChild(size_t idx) const->Field;
      void setChild(size_t idx, const Field& child);
      void addChild(const Field& field);
      void popChild(size_t idx);
    private:
      friend struct Field;
      FieldRef(const std::shared_ptr<type>& object, size_t idx) :
        impl_type(impl::ObjectWrapperHolderChildObjectRef<FieldObject>(object, idx))
      {}
    };

    struct FieldSerializer : public ObjectSerializer {
      virtual ~FieldSerializer() noexcept;
      auto getTypeString() const noexcept -> Str override;
      auto eval(const std::shared_ptr<Object>& object) const->Json override;
    };
  }
}
