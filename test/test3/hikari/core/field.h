#pragma once
#include <hikari/core/object.h>
#include <hikari/core/json.h>
namespace hikari {
  inline namespace core {
    struct FieldObject : public Object {
      using BaseType = Object;
      static inline Bool Convertible(const Str& str) noexcept {
        if (BaseType::Convertible(str)) { return true; }
        if (str == TypeString()) { return true; }
        return false;
      }
      static inline constexpr auto TypeString() -> const char* { return "Field"; }
      static auto create(Str name = "") -> std::shared_ptr<FieldObject>;
      virtual ~FieldObject() noexcept;

      virtual auto getTypeString() const -> Str override { return TypeString(); }
      virtual auto getJSONString() const->Str override;

      virtual auto getName() const->Str override;
      void setName(const Str& name);

      virtual auto getPropertyNames() const->std::vector<Str> override;
      virtual void setPropertyBlock(const PropertyBlock& pb) override;
      virtual void getPropertyBlock(PropertyBlock& pb) const override;
      virtual Bool hasProperty(const Str& name) const override;
      virtual Bool setProperty(const Str& name, const Property& value) override;
      virtual Bool getProperty(const Str& name, Property& value) const override;
      virtual Bool isConvertible(const Str& type_name) const override
      {
        return Convertible(type_name);
      }

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
    struct Field {
      using TypesTuple = PropertyTypes;
    public:
      using ObjectType = FieldObject;
      template<typename T>
      using Traits = in_tuple<T, TypesTuple>;
      using PropertyRef = ObjectPropertyRef;

      Field() noexcept :m_object{} {}
      Field(nullptr_t) noexcept :m_object{ nullptr } {}
      Field(const std::string& name) noexcept : m_object{ ObjectType::create(name) } {}
      Field(const Field&) = default;
      Field& operator=(const Field&) = default;
      Field(const std::shared_ptr<ObjectType>& object) :m_object{ object } {}
      Field& operator=(const std::shared_ptr<ObjectType>& obj) { m_object = obj; return *this; }

      Bool operator!() const noexcept { return !m_object; }
      operator Bool () const noexcept { return m_object != nullptr; }

      template<size_t N>
      auto operator[](const char(&name)[N])->PropertyRef { return operator[](Str(name)); }
      template<size_t N>
      auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
      auto operator[](const Str& name)->PropertyRef;
      auto operator[](const Str& name) const->Property;
      auto operator[](size_t idx) const->Field;
      auto operator[](size_t idx)->FieldRef;

      auto clone() -> Field;

      void setPropertyBlock(const PropertyBlock& pb);
      void getPropertyBlock(PropertyBlock& pb) const;

      auto getJSONString() const -> std::string {
        auto object = getObject();
        if (object) { return object->getJSONString(); }
        else { return "null"; }
      }

      auto getSize() const->size_t;
      void setSize(size_t     count);

      auto getName() const->Str;
      void setName(const Str& name);

      auto getObject() const->std::shared_ptr<ObjectType>;
      auto getKeys() const->std::vector<Str>;
      Bool getTypeIndex(const Str& name, size_t& type_index) const {
        if (name == "children") { return false; }
        return m_object->getPropertyTypeIndex(name, type_index);
      }

      Bool setValue(const Str& name, const Property& prop);
      Bool getValue(const Str& name, Property& prop) const;
      auto getValue(const Str& name) const->Property;
      Bool hasValue(const Str& name) const;

      template<typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
      void setValue(const Str& name, T value) noexcept { return setValue(name, Property(value)); }
      template<size_t N>
      void setValue(const Str& name, const char(&value)[N]) noexcept { setValue(name, Str(value)); }

      auto getChildCount() const->size_t;
      void setChildCount(size_t count);

      auto getChildren() const->std::vector<Field>;
      void setChildren(const std::vector<Field>& children);
      void popChildren();

      auto getChild(size_t idx) const->Field;
      void setChild(size_t idx, const Field& child);
      void addChild(const Field& field);
      void popChild(size_t idx);
    private:
      std::shared_ptr<ObjectType> m_object;
    };
    struct FieldRef {
      using ObjectType = FieldObject;
      using Ref = FieldRef;
      using PropertyRef = Field::PropertyRef;
      template<typename T>
      using Traits = Field::Traits<T>;

      FieldRef(const FieldRef&) noexcept = delete;
      FieldRef(FieldRef&&) noexcept = delete;
      FieldRef& operator=(const FieldRef&) = delete;
      FieldRef& operator=(FieldRef&&) = delete;

      void operator=(const Field& field) noexcept;
      operator Field() const { return Field(getObject()); }

      template<size_t N>
      auto operator[](const char(&name)[N])->PropertyRef { return operator[](Str(name)); }
      template<size_t N>
      auto operator[](const char(&name)[N])const ->Property { return operator[](Str(name)); }
      auto operator[](const Str& name)->PropertyRef;
      auto operator[](const Str& name) const->Property;
      auto operator[](size_t idx) const->Field;
      auto operator[](size_t idx)->Ref;

      void setPropertyBlock(const PropertyBlock& pb);
      void getPropertyBlock(PropertyBlock& pb) const;

      auto getSize() const->size_t;
      void setSize(size_t     count);

      auto getName() const->Str;
      void setName(const Str& name);

      auto getObject() const->std::shared_ptr<ObjectType>;
      auto getKeys() const->std::vector<Str>;

      Bool setValue(const Str& name, const Property& prop);
      Bool getValue(const Str& name, Property& prop) const;
      auto getValue(const Str& name) const->Property;
      Bool hasValue(const Str& name) const;

      template<typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
      void setValue(const Str& name, T value) noexcept { return setValue(name, Property(value)); }
      template<size_t N>
      void setValue(const Str& name, const char(&value)[N]) noexcept { setValue(name, Str(value)); }

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
      FieldRef(const std::shared_ptr<ObjectType>& object, size_t idx) :m_object{ object }, m_idx{ idx } {}
      std::weak_ptr<ObjectType> m_object;
      size_t m_idx;
    };

    auto convertJSONToField(const Json& json) -> Field;
    auto convertFieldToJSON(const Field& field) -> Json;

    auto convertJSONStringToField(const Str& str) -> Field;

    template<>
    struct ConvertFromJSONStringTraits<Field> :std::true_type {
      static auto eval(const Str& str) -> Option<Field> {
        auto field = convertJSONStringToField(str);
        if (!field) { return std::nullopt; }
        return field;
      }
    };

    auto convertToJSONString(const core::Field& v) -> Str;

  }
}
