#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <type_traits>
#include <string_view>
#include <variant>
#include <hikari/core/types/object.h>
#include <hikari/core/types/property.h>
#include <hikari/core/types/ref_delagate_def.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif
    struct Properties : public Object {
      HK_CORE_TYPES_OBJECT_DEFINE_TYPE_RELATIONSHIP(Properties, Object);
      virtual ~Properties() noexcept {}
      virtual auto getPropertyNames() const-> ArrayString = 0;
      virtual Bool hasProperty(const String& name) const  = 0;
      virtual Bool getProperty(const String& name, Property& prop) const = 0;
      virtual Bool setProperty(const String& name, const Property& prop) = 0;
      inline  auto getProperty(const String& name) const-> Property {
        Property prop;
        if (getProperty(name, prop)) { return prop; }
        return Property();
      }
    };

    struct WRefPropertyHolderForRefProperties {
      WRefPropertyHolderForRefProperties(const std::shared_ptr<Properties>& p, const String& name) noexcept
        : m_properties{ p }, m_name{ name } {}
      auto getProperty() const -> Property {
        auto ref = m_properties.lock();
        if (ref) {
          return ref->getProperty(m_name);
        }
        else {
          return Property();
        }
      }
      void setProperty(const Property& p) {
        auto ref = m_properties.lock();
        if (ref) {
          ref->setProperty(m_name,p);
        }
      }
    private:
      std::weak_ptr<Properties> m_properties;
      std::string m_name;
    };

    using WRefPropertyForRefProperties = WRefProperty<WRefPropertyHolderForRefProperties>;

    template<typename ObjectOwnerT, typename ObjectReturnT, template<typename ObjectOwnerT, typename ObjectReturnT> typename ObjectHolder>
    struct  RefPropertiesBase : protected RefObjectBase<ObjectOwnerT, ObjectReturnT,ObjectHolder>{
      using impl_type = RefObjectBase<ObjectOwnerT, ObjectReturnT, ObjectHolder>;
      HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS();
      RefPropertiesBase() noexcept : impl_type() {}
      HK_CORE_TYPES_REF_OBJECT_BASE_DEF_CONS_AND_DEST(RefPropertiesBase);
      HK_CORE_TYPES_REF_OBJECT_USING_METHODS();
      HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD(0, 1, 1, DEF)(getRef(), ArrayString, getPropertyNames, ArrayString());
      HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD(1, 1, 1, DEF)(getRef(), Bool, hasProperty, const String&, name, false);
      HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD(2, 1, 1, DEF)(getRef(), Bool, getProperty, const String&, name, Property&, prop, false);
      HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD(2, 1, 0, DEF)(getRef(), Bool, setProperty, String&, name, const Property&, prop, false);
      auto operator[](const String& name)   const -> Property { return getProperty(name); }
      template<UPtr N>
      auto operator[](const Char(&name)[N]) const -> Property { return getProperty(name); }
      auto operator[](const String& name)   -> WRefPropertyForRefProperties { return WRefPropertyForRefProperties(WRefPropertyHolderForRefProperties(getRef(), name)); }
      template<UPtr N>
      auto operator[](const Char(&name)[N]) -> WRefPropertyForRefProperties { return WRefPropertyForRefProperties(WRefPropertyHolderForRefProperties(getRef(), name)); }
    };
#define HK_CORE_TYPES_REF_PROPERTIES_USING_METHODS() \
      using impl_type::getPropertyNames; \
      using impl_type::hasProperty; \
      using impl_type::getProperty; \
      using impl_type::setProperty; \
      using impl_type::operator[]
    
    struct  SRefProperties : protected RefPropertiesBase<Properties, Properties, SRefObjectHolder> {
      using impl_type = RefPropertiesBase<Properties, Properties, SRefObjectHolder>;
      HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS();
      SRefProperties() noexcept : impl_type() {}
      HK_CORE_TYPES_REF_OBJECT_DEF_TO_BOOLEAN(SRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_SHARED(SRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_REFOBJ(SRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_COPY_AND_MOVE(SRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_COMPARISON(SRefProperties);
      HK_CORE_TYPES_REF_OBJECT_USING_METHODS();
      HK_CORE_TYPES_REF_PROPERTIES_USING_METHODS();
    };
    struct  WRefProperties : protected RefPropertiesBase<Properties, Properties, WRefObjectHolder> {
      using impl_type = RefPropertiesBase<Properties, Properties, WRefObjectHolder>;
      HK_CORE_TYPES_REF_OBJECT_TYPE_ALIAS();
      WRefProperties() noexcept : impl_type() {}
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_SHARED(WRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_FROM_REFOBJ(WRefProperties);
      HK_CORE_TYPES_REF_OBJECT_DEF_COPY_AND_MOVE(WRefProperties);
      HK_CORE_TYPES_REF_OBJECT_USING_METHODS();
      HK_CORE_TYPES_REF_PROPERTIES_USING_METHODS();
    };

#if defined(__cplusplus)
  }
}
#endif
