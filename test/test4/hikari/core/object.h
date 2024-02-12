#pragma once
#include <memory>
#include <hikari/core/property.h>

#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_SETTER_LIKE(METHOD, ARG) \
  void METHOD(const ARG& arg) { auto object = getObject(); if (object){ object->METHOD(arg); } }
#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE(METHOD, RES) \
  Option<RES> METHOD() const { auto object = getObject(); if (object){ return RES(object->METHOD()); } else { return std::nullopt;} }
#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_DEF(METHOD, RES,DEF) \
  RES METHOD() const { auto object = getObject(); if (object){ return RES(object->METHOD()); } else { return RES(DEF);} }
#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_OPTION(METHOD, RES) \
  Option<RES> METHOD() const { auto object = getObject(); if (object){ return object->METHOD(); } else { return std::nullopt;} }
#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK(METHOD, RES) \
  Bool METHOD(RES& res) const { auto object = getObject(); if (object){ return object->METHOD(res); } return false; }
#define HK_OBJECT_WRAPPER_METHOD_OVERLOAD_GETTER_LIKE_WITH_CHECK_FROM_VOID(METHOD, RES) \
  Bool METHOD(RES& res) const { auto object = getObject(); if (object){ object->METHOD(res); return true; } return false; }

namespace hikari {
  inline namespace core {
    template<typename ObjectT>
    struct ObjectPropertyRefBase
    {
      using data_types                  = typename PropertyCommonDefinitionsBase<ObjectT>::data_types;
      using array_data_types            = typename PropertyCommonDefinitionsBase<ObjectT>::array_data_types;
      using types                       = typename PropertyCommonDefinitionsBase<ObjectT>::types;
      template<typename T> using traits_data_types           = typename PropertyTraitsBase<ObjectT, T>::traits_data_types;
      template<typename T> using traits_array_data_types     = typename PropertyTraitsBase<ObjectT, T>::traits_array_data_types;
      template<typename T> using traits_object_wrapper       = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper;
      template<typename T> using traits_object_wrapper_array = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper_array;
      static inline constexpr size_t kTypeSwitchIdxDataTypes          = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDataTypes;
      static inline constexpr size_t kTypeSwitchIdxArrayDataTypes     = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxArrayDataTypes;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapper      = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapper;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapperArray = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapperArray;
      static inline constexpr size_t kTypeSwitchIdxDefault            = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDefault;
    private:
      template<typename T, size_t idx = PropertyTraitsBase<ObjectT,T>::kTypeSwitchIdx>
      struct impl_type_switch : std::false_type {};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxDataTypes>          : std::true_type { using result_type = Option<T>;};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxArrayDataTypes>     : std::true_type { using result_type = T;};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapper>      : std::true_type { using result_type = T; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapperArray> : std::true_type { using result_type = T; };
    public:
      ObjectPropertyRefBase(const std::shared_ptr<ObjectT>& object, const Str& name) : m_holder{ object }, m_name{ name }{}

      ObjectPropertyRefBase(const ObjectPropertyRefBase&) noexcept = delete;
      ObjectPropertyRefBase(ObjectPropertyRefBase&&) noexcept = delete;
      ObjectPropertyRefBase& operator=(const ObjectPropertyRefBase&) = delete;
      ObjectPropertyRefBase& operator=(ObjectPropertyRefBase&&) = delete;

      void operator=(const PropertyBase<ObjectT>& prop) noexcept { auto pb = m_holder.lock(); if (!pb) { return; } pb->setProperty(m_name, prop); }
      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      void operator=(const T& value) noexcept { auto pb = m_holder.lock(); if (pb) { pb->setProperty(m_name, PropertyBase<ObjectT>(value)); }; }
      template <size_t N>
      void operator=(const char(&value)[N]) noexcept
      {
        auto pb = m_holder.lock();
        pb->setProperty(m_name, Property(Str(value)));
      }

      operator PropertyBase<ObjectT>() const noexcept
      {
        auto pb = m_holder.lock();
        if (!pb)
        {
          return PropertyBase<ObjectT>();
        }
        PropertyBase<ObjectT> res;
        if (!pb->getProperty(m_name, res))
        {
          return PropertyBase<ObjectT>();
        }
        return res;
      }
      explicit operator Bool() const noexcept
      {
        auto pb = m_holder.lock();
        if (!pb)
        {
          return false;
        }
        if (!pb->hasProperty(m_name))
        {
          return false;
        }
        return true;
      }

      Bool hasValue() const noexcept {
        auto pb = m_holder.lock();
        if (!pb) { return false; }
        return pb->hasProperty(m_name);
      }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      auto getValue() const noexcept -> typename impl_type_switch<T>::result_type {
        auto pb = m_holder.lock();
        if (!pb) { return {}; }
        PropertyBase<ObjectT> res;
        if (!pb->getProperty(m_name, res)) { return {}; }
        return res.getValue<T>();
      }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      bool setValue(const T& value) noexcept {
        auto pb = m_holder.lock();
        if (!pb) { return false; };
        return pb->setProperty(m_name, PropertyBase<ObjectT>(value));
      }
    private:
      std::weak_ptr<ObjectT> m_holder;
      Str m_name;
    };

    namespace impl {
      template<typename ObjectDerive>
      struct ObjectWrapperHolderSharedRef {
        ObjectWrapperHolderSharedRef() noexcept : m_ref{ nullptr } {}
        ObjectWrapperHolderSharedRef(const ObjectWrapperHolderSharedRef&) = default;
        ObjectWrapperHolderSharedRef& operator=(const ObjectWrapperHolderSharedRef&) = default;
        ObjectWrapperHolderSharedRef(const std::shared_ptr<ObjectDerive>& ptr) noexcept : m_ref{ ptr } {}
        ObjectWrapperHolderSharedRef& operator=(const std::shared_ptr<ObjectDerive>& ptr) noexcept { m_ref = ptr; return *this; }

        auto getObject() const noexcept -> std::shared_ptr<ObjectDerive> { return m_ref; }
      protected:
        void setObject(const std::shared_ptr<ObjectDerive>& ptr) noexcept { m_ref = ptr; }
      private:
        std::shared_ptr<ObjectDerive> m_ref;
      };
      template<typename ObjectDerive>
      struct ObjectWrapperHolderWeakRef {
        ObjectWrapperHolderWeakRef() noexcept : m_ref{  } {}
        ObjectWrapperHolderWeakRef(const ObjectWrapperHolderWeakRef&) = default;
        ObjectWrapperHolderWeakRef& operator=(const ObjectWrapperHolderWeakRef&) = default;
        ObjectWrapperHolderWeakRef(const std::shared_ptr<ObjectDerive>& ptr) noexcept : m_ref{ ptr } {}
        ObjectWrapperHolderWeakRef& operator=(const std::shared_ptr<ObjectDerive>& ptr) noexcept { m_ref = ptr; return *this; }

        auto getObject() const noexcept -> std::shared_ptr<ObjectDerive> { return m_ref.lock(); }
      protected:
        void setObject(const std::shared_ptr<ObjectDerive>& ptr) noexcept { m_ref = ptr; }
      private:
        std::weak_ptr<ObjectDerive> m_ref;
      };
      template<typename ObjectT>
      struct ObjectWrapperHolderChildObjectRef {
        using wrapper_type = typename ObjectT::wrapper_type;

        ObjectWrapperHolderChildObjectRef(const std::shared_ptr<ObjectT>& object, size_t idx) noexcept : m_object{ object }, m_idx{ idx } {}
        ObjectWrapperHolderChildObjectRef(const ObjectWrapperHolderChildObjectRef&)noexcept = default;
        ObjectWrapperHolderChildObjectRef& operator=(const ObjectWrapperHolderChildObjectRef&)noexcept = default;
        ObjectWrapperHolderChildObjectRef& operator=(const std::shared_ptr<ObjectT>& ptr)noexcept {
          auto object = m_object.lock();
          if (!object) { return *this; }
          object->setChild(m_idx, ptr);
          return *this;
        }

        auto getObject() const noexcept -> std::shared_ptr<ObjectT> {
          auto object = m_object.lock();
          if (!object) { return nullptr; }
          Property prop;
          if (!object->getProperty("children", prop)) { return nullptr; }
          auto children = prop.getValue<Array<wrapper_type>>();
          if (children.size() <= m_idx) { return nullptr; }
          else { return children[m_idx].getObject(); }
        }
      private:
        std::weak_ptr<ObjectT> m_object;
        size_t m_idx;
      };
    }

    template <typename ObjectT, template <typename...> typename ObjectHolder, typename ObjectDerive, std::enable_if_t<std::is_base_of_v<ObjectT, ObjectDerive>, nullptr_t> = nullptr>
    struct ObjectWrapperImplBase
    {
      using data_types                  = typename PropertyCommonDefinitionsBase<ObjectT>::data_types;
      using array_data_types            = typename PropertyCommonDefinitionsBase<ObjectT>::array_data_types;
      using types                       = typename PropertyCommonDefinitionsBase<ObjectT>::types;
      template<typename T> using traits_data_types           = typename PropertyTraitsBase<ObjectT, T>::traits_data_types;
      template<typename T> using traits_array_data_types     = typename PropertyTraitsBase<ObjectT, T>::traits_array_data_types;
      template<typename T> using traits_object_wrapper       = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper;
      template<typename T> using traits_object_wrapper_array = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper_array;
      static inline constexpr size_t kTypeSwitchIdxDataTypes          = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDataTypes;
      static inline constexpr size_t kTypeSwitchIdxArrayDataTypes     = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxArrayDataTypes;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapper      = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapper;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapperArray = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapperArray;
      static inline constexpr size_t kTypeSwitchIdxDefault            = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDefault;
    private:
      template<typename T, size_t idx = PropertyTraitsBase<ObjectT,T>::kTypeSwitchIdx>
      struct impl_type_switch : std::false_type {};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxDataTypes>          : std::true_type { using result_type = Option<T>;};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxArrayDataTypes>     : std::true_type { using result_type = T; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapper>      : std::true_type { using result_type = T; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapperArray> : std::true_type { using result_type = T; };
    public:
      using property_type     = PropertyBase<ObjectT>;
      using property_ref_type = ObjectPropertyRefBase<ObjectT>;
      using type              = ObjectDerive;

      ObjectWrapperImplBase() noexcept : m_holder{} {}
      ObjectWrapperImplBase(nullptr_t) noexcept : m_holder{ nullptr } {}
      ObjectWrapperImplBase(const ObjectWrapperImplBase&) = default;
      ObjectWrapperImplBase(const std::shared_ptr<type>& object) : m_holder{ object } {}

      auto getPropertyNames() const -> std::vector<Str>
      {
        auto object = getObject();
        if (!object)
        {
          return {};
        }
        return object->getPropertyNames();
      }

      auto operator[](const Str& name)       -> property_ref_type  { auto object = getObject(); return property_ref_type(object, name); }
      auto operator[](const Str& name) const -> property_type      { return getValue(name); }
      template <size_t N>
      auto operator[](const char(&name)[N]) -> property_ref_type   { return operator[](Str(name)); }
      template <size_t N>
      auto operator[](const char(&name)[N]) const -> property_type { return operator[](Str(name)); }

      Bool operator!() const noexcept
      {
        auto object = getObject();
        return !object;
      }
      operator Bool() const noexcept
      {
        auto object = getObject();
        return object != nullptr;
      }

      auto getName() const -> Str { auto pb = getObject(); if (!pb) { return ""; } return pb->getName(); }
      Bool isConvertible(const Str& type) const noexcept {
        auto object = getObject();
        if (!object) { return false; }
        return object->isConvertible(type);
      }

      void getPropertyBlock(PropertyBlockBase<ObjectT>& pb) const
      {
        auto object = getObject();
        if (!object)
        {
          return;
        }
        object->getPropertyBlock(pb);
      }
      void setPropertyBlock(const PropertyBlockBase<ObjectT>& pb)
      {
        auto object = getObject();
        if (!object)
        {
          return;
        }
        object->setPropertyBlock(pb);
      }

      Bool setValue(const Str& name, const property_type& prop)
      {
        auto object = getObject();
        if (!object)
        {
          return false;
        }
        return object->setProperty(name, prop);
      }
      Bool getValue(const Str& name, property_type& prop) const
      {
        auto object = getObject();
        if (!object)
        {
          return false;
        }
        return object->getProperty(name, prop);
      }
      auto getValue(const Str& name) const -> property_type
      {
        auto object = getObject();
        if (!object)
        {
          return Property();
        }
        property_type res;
        if (!object->getProperty(name,res)) { return {}; }
        return res;
      }
      Bool hasValue(const Str& name) const
      {
        auto object = getObject();
        if (!object)
        {
          return false;
        }
        return object->hasProperty(name);
      }

      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      Bool setValue(const Str& name, const T& value) { return setValue(name, property_type(value)); }
      template <size_t N>
      Bool setValue(const Str& name, const char(&value)[N])  { return setValue(name, Str(value)); }

      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      auto getValue(const Str& name) const -> typename impl_type_switch<T>::result_type
      {
        auto object = getObject();
        if (!object)
        {
          return {};
        }
        property_type res;
        if (!object->getProperty(name, res)) { return {}; }
        return res.getValue<T>();
      }

      auto getObject() const -> std::shared_ptr<ObjectDerive> { return m_holder.getObject(); }
    protected:
      void setObject(const std::shared_ptr<ObjectDerive>& object) { m_holder = object; }
    private:
      ObjectHolder<ObjectDerive> m_holder;
    };

    template<typename ObjectT>
    struct ObjectWrapperBase : protected ObjectWrapperImplBase<ObjectT,impl::ObjectWrapperHolderSharedRef,ObjectT>{
      using impl_type         = ObjectWrapperImplBase<ObjectT, impl::ObjectWrapperHolderSharedRef, ObjectT>;
      using property_ref_type = typename impl_type::property_ref_type;
      using property_type     = typename impl_type::property_type;
      using type              = typename impl_type::type;

      ObjectWrapperBase() noexcept : impl_type() {}
      ObjectWrapperBase(nullptr_t) noexcept : impl_type(nullptr) {}
      ObjectWrapperBase(const std::shared_ptr<ObjectT>& object) noexcept : impl_type(object) {}
      ObjectWrapperBase(const ObjectWrapperBase& opb) noexcept : impl_type(opb.getObject()) {}
      ObjectWrapperBase(ObjectWrapperBase&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      ObjectWrapperBase& operator=(const ObjectWrapperBase& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      ObjectWrapperBase& operator=(ObjectWrapperBase&& opb) noexcept
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
      ObjectWrapperBase& operator=(const std::shared_ptr<ObjectT>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<ObjectT, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      ObjectWrapperBase(const ObjectWrapperLike& wrapper) noexcept : impl_type(wrapper.getObject()) {}
      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<ObjectT, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      ObjectWrapperBase& operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      
      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::operator[];
      using impl_type::isConvertible;
      using impl_type::getName;
      using impl_type::getPropertyNames;
      using impl_type::getObject;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getValue;
      using impl_type::hasValue;
      using impl_type::setValue;
    };

    struct Object {
      static Bool Convertible(const Str& type) noexcept {
        if (type == TypeString()) { return true; }
        return false;
      }
      static constexpr const char* TypeString() { return "Object"; };
      virtual ~Object() noexcept {}

      virtual auto getName() const -> Str { return ""; }

      virtual Str  getTypeString() const noexcept = 0;
      virtual Bool isConvertible(const Str& type) const noexcept = 0;

      virtual auto getPropertyNames() const -> std::vector<Str> = 0;

      virtual void getPropertyBlock(PropertyBlockBase<Object>& pb) const = 0;
      virtual void setPropertyBlock(const PropertyBlockBase<Object>& pb) = 0;

      virtual Bool hasProperty(const Str& name) const = 0;
      virtual Bool getProperty(const Str& name, PropertyBase<Object>& prop) const = 0;
      virtual Bool setProperty(const Str& name, const PropertyBase<Object>& prop) = 0;
    };

    namespace ObjectUtils {
      template<typename ObjectTTo, typename ObjectTFrom>
      auto convert(const std::shared_ptr<ObjectTFrom>& from) -> decltype(std::enable_if_t<
        std::is_base_of_v<Object, ObjectTFrom>&&
        std::is_base_of_v<Object, ObjectTTo  >&&
        std::is_base_of_v<ObjectTFrom, ObjectTTo  >, nullptr_t>{nullptr},
        std::shared_ptr<ObjectTTo>()) {
        if (!from) { return nullptr; }
        if (!from->isConvertible(ObjectTTo::TypeString())) { return nullptr;  }
        return std::static_pointer_cast<ObjectTTo>(from);
      }
      template<typename ObjectTTo, typename ObjectTFrom>
      auto convert(const std::shared_ptr<ObjectTFrom>& from) -> decltype(std::enable_if_t<
        std::is_base_of_v<Object, ObjectTFrom>&&
        std::is_base_of_v<Object, ObjectTTo  >&&
        std::is_base_of_v<ObjectTTo, ObjectTFrom  >, nullptr_t>{nullptr},
        std::shared_ptr<ObjectTTo>()) {
        return std::static_pointer_cast<ObjectTTo>(from);
      }
    }
    namespace ObjectWrapperUtils {
      template<typename ObjectTTo, typename ObjectTFrom>
      auto convert(const ObjectTFrom& from) -> decltype(std::enable_if_t <
        std::is_base_of_v<Object, typename ObjectTFrom::type> &&
        std::is_base_of_v<Object, typename ObjectTTo::type  >&&
        !std::is_same_v<typename ObjectTFrom::type, typename ObjectTTo::type> &&
        std::is_base_of_v<typename ObjectTFrom::type, typename ObjectTTo::type>, nullptr_t>{nullptr},
        ObjectTTo()) {
        auto from_object = from.getObject();
        auto to_object = ObjectUtils::convert<typename ObjectTTo::type>(from_object);
        return ObjectTTo(to_object);
      }
      template<typename ObjectTTo, typename ObjectTFrom>
      auto convert(const ObjectTFrom& from) -> decltype(std::enable_if_t <
        std::is_base_of_v < Object, typename ObjectTFrom::type>&&
        std::is_base_of_v<Object, typename ObjectTTo::type  >&&
        !std::is_same_v<typename ObjectTFrom::type, typename ObjectTTo::type> &&
        std::is_base_of_v<typename ObjectTTo::type, typename ObjectTFrom::type>, nullptr_t>{nullptr},
        ObjectTTo()) {
        auto from_object = from.getObject();
        auto to_object = ObjectUtils::convert<typename ObjectTTo::type>(from_object);
        return ObjectTTo(to_object);
      }
      template<typename ObjectTTo, typename ObjectTFrom>
      auto convert(const ObjectTFrom& from) -> decltype(std::enable_if_t <
        std::is_base_of_v < Object, typename ObjectTFrom::type>&&
        std::is_base_of_v<Object, typename ObjectTTo::type  > &&
        std::is_same_v<typename ObjectTFrom::type, typename ObjectTTo::type>, nullptr_t>{nullptr},
        ObjectTTo()) {
        auto from_object = from.getObject();
        return ObjectTTo(from_object);
      }
    }

    template <typename ObjectT, template <typename...> typename ObjectHolder, typename ObjectDerive, std::enable_if_t<std::is_base_of_v<ObjectT, ObjectDerive>, nullptr_t> = nullptr>
    struct ObjectWrapperRefImplBase
    {
      using data_types = typename PropertyCommonDefinitionsBase<ObjectT>::data_types;
      using array_data_types = typename PropertyCommonDefinitionsBase<ObjectT>::array_data_types;
      using types = typename PropertyCommonDefinitionsBase<ObjectT>::types;
      template<typename T> using traits_data_types = typename PropertyTraitsBase<ObjectT, T>::traits_data_types;
      template<typename T> using traits_array_data_types = typename PropertyTraitsBase<ObjectT, T>::traits_array_data_types;
      template<typename T> using traits_object_wrapper = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper;
      template<typename T> using traits_object_wrapper_array = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper_array;
      static inline constexpr size_t kTypeSwitchIdxDataTypes = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDataTypes;
      static inline constexpr size_t kTypeSwitchIdxArrayDataTypes = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxArrayDataTypes;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapper = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapper;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapperArray = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapperArray;
      static inline constexpr size_t kTypeSwitchIdxDefault = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDefault;
    private:
      template<typename T, size_t idx = PropertyTraitsBase<ObjectT, T>::kTypeSwitchIdx>
      struct impl_type_switch : std::false_type {};
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxDataTypes> : std::true_type { using result_type = Option<T>; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxArrayDataTypes> : std::true_type { using result_type = T; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapper> : std::true_type { using result_type = T; };
      template<typename T> struct impl_type_switch<T, kTypeSwitchIdxObjectWrapperArray> : std::true_type { using result_type = T; };
    public:
      using property_type     = PropertyBase<ObjectT>;
      using property_ref_type = ObjectPropertyRefBase<ObjectT>;
      using type              = ObjectDerive;
      using wrapper_type      = typename ObjectHolder<ObjectDerive>::wrapper_type;

      ObjectWrapperRefImplBase() noexcept : m_holder{} {}
      ObjectWrapperRefImplBase(nullptr_t) noexcept : m_holder{ nullptr } {}
      ObjectWrapperRefImplBase(const ObjectHolder<type>& holder) : m_holder{ holder } {}

      ObjectWrapperRefImplBase(const ObjectWrapperRefImplBase&) noexcept   = delete;
      ObjectWrapperRefImplBase(ObjectWrapperRefImplBase&&) noexcept        = delete;
      ObjectWrapperRefImplBase& operator=(const ObjectWrapperRefImplBase&) = delete;
      ObjectWrapperRefImplBase& operator=(ObjectWrapperRefImplBase&&)      = delete;

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<ObjectT, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      void operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
      }

      void operator=(const wrapper_type& wrapper) noexcept {
        setObject(wrapper.getObject());
      }
      operator wrapper_type() const { return wrapper_type(getObject()); }

      auto getPropertyNames() const -> std::vector<Str>
      {
        auto object = getObject();
        if (!object)
        {
          return {};
        }
        return object->getPropertyNames();
      }

      auto operator[](const Str& name)       -> property_ref_type { auto object = getObject(); return property_ref_type(object, name); }
      auto operator[](const Str& name) const -> property_type { return getValue(name); }
      template <size_t N>
      auto operator[](const char(&name)[N]) -> property_ref_type { return operator[](Str(name)); }
      template <size_t N>
      auto operator[](const char(&name)[N]) const -> property_type { return operator[](Str(name)); }

      Bool operator!() const noexcept
      {
        auto object = getObject();
        return !object;
      }
      operator Bool() const noexcept
      {
        auto object = getObject();
        return object != nullptr;
      }

      auto getName() const -> Str { auto pb = getObject() if (!pb) { return ""; } return pb->getName(); }
      Bool isConvertible(const Str& type) const noexcept {
        auto object = getObject();
        if (!object) { return false; }
        return object->isConvertible(type);
      }

      void getPropertyBlock(PropertyBlockBase<ObjectT>& pb) const
      {
        auto object = getObject();
        if (!object)
        {
          return;
        }
        object->getPropertyBlock(pb);
      }
      void setPropertyBlock(const PropertyBlockBase<ObjectT>& pb)
      {
        auto object = getObject();
        if (!object)
        {
          return;
        }
        object->setPropertyBlock(pb);
      }

      Bool setValue(const Str& name, const property_type& prop)
      {
        auto object = getObject();
        if (!object)
        {
          return;
        }
        return object->setProperty(name, prop);
      }
      Bool getValue(const Str& name, property_type& prop) const
      {
        auto object = getObject();
        if (!object)
        {
          return false;
        }
        return object->getProperty(name, prop);
      }
      auto getValue(const Str& name) const -> property_type
      {
        auto object = getObject();
        if (!object)
        {
          return Property();
        }
        property_type res;
        if (!object->getProperty(name, res)) { return {}; }
        return res;
      }
      Bool hasValue(const Str& name) const
      {
        auto object = getObject();
        if (!object)
        {
          return false;
        }
        return object->hasProperty(name);
      }

      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      Bool setValue(const Str& name, const T& value) { return setValue(name, property_type(value)); }
      template <size_t N>
      Bool setValue(const Str& name, const char(&value)[N]) { return setValue(name, Str(value)); }

      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      auto getValue(const Str& name) const -> typename impl_type_switch<T>::result_type
      {
        auto object = getObject();
        if (!object)
        {
          return {};
        }
        property_type res;
        if (!object->getProperty(name, res)) { return {}; }
        return res.getValue<T>();
      }

      auto getObject() const -> std::shared_ptr<ObjectDerive> { return m_holder.getObject(); }
    protected:
      void setObject(const std::shared_ptr<ObjectDerive>& object) { m_holder = object; }
    private:
      ObjectHolder<ObjectDerive> m_holder;
    };

    using ObjectWrapper       = ObjectWrapperBase<Object>;
    template <template <typename...> typename ObjectHolder, typename ObjectDerive>
    using ObjectWrapperImpl   = ObjectWrapperImplBase<Object, ObjectHolder, ObjectDerive>;
    template <template <typename...> typename ObjectHolder, typename ObjectDerive>
    using ObjectWrapperRefImpl= ObjectWrapperRefImplBase<Object, ObjectHolder, ObjectDerive>;
    template<typename T>
    using ObjectWrapperTraits = ObjectWrapperTraitsBase<Object, T>;
    template<typename T>
    using ObjectWrapperArrayTraits = ObjectWrapperArrayTraitsBase<Object, T>;
    using Property            = PropertyBase<Object>;
    using PropertyBlock       = PropertyBlockBase<Object>;
    using PropertySingleTypes = PropertySingleTypesBase<Object>;
    using PropertyArrayTypes  = PropertyArrayTypesBase<Object>;
    using PropertyTypes       = PropertyTypesBase<Object>;
    template<typename T>
    using PropertyTypeIndex   = find_tuple<T, PropertyTypesBase<Object>>;

    HK_TYPE_2_STRING_DEFINE(Object);
    HK_TYPE_2_STRING_DEFINE(ObjectWrapper);
    HK_TYPE_2_STRING_DEFINE(Property);
    HK_TYPE_2_STRING_DEFINE(PropertyBlock);
  }
}
