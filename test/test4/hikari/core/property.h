#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <type_traits>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <hikari/core/data_type.h>
#include <hikari/core/transform.h>
#include <hikari/core/tuple.h>
#include <hikari/core/utils.h>

namespace hikari {
  inline namespace core {
    // 任意精度のPropertyを取得可能
    using PropertySignedTypes    = std::tuple<hikari::I8 , hikari::I16, hikari::I32, hikari::I64>;
    using PropertyUnsignedTypes  = std::tuple<hikari::U8 , hikari::U16, hikari::U32, hikari::U64>;
    using PropertyFloatTypes     = std::tuple<hikari::F32, hikari::F64>;
    using PropertyIntegerTypes   = typename concat_tuple<PropertySignedTypes, PropertyUnsignedTypes>::type;
    using PropertyNumericTypes   = typename concat_tuple<PropertySignedTypes, PropertyUnsignedTypes, PropertyFloatTypes, std::tuple<hikari::Bool>>::type;
    using PropertyVectorTypes    = std::tuple<Vec2, Vec3, Vec4>;
    using PropertyMatrixTypes    = std::tuple<Mat2, Mat3, Mat4>;
    using PropertyDataTypes      = typename concat_tuple<PropertyNumericTypes, PropertyVectorTypes, PropertyMatrixTypes, std::tuple<Quat,Transform,Str>>::type;
    using PropertyArrayDataTypes = typename transform_tuple<Array, PropertyDataTypes>::type;

    template<typename ObjectT>
    using PropertySingleTypesBase = typename concat_tuple<
      PropertyDataTypes,
      std::tuple<std::shared_ptr<ObjectT>>
    >::type;
    template<typename ObjectT>
    using PropertyArrayTypesBase  = typename transform_tuple<Array, PropertySingleTypesBase<ObjectT>>::type;
    template<typename ObjectT>
    using PropertyTypesBase       = typename concat_tuple<PropertySingleTypesBase<ObjectT>, PropertyArrayTypesBase<ObjectT>>::type;
    template<typename ObjectT,typename T>
    using PropertyTypeIndexBase   = find_tuple<T, PropertyTypesBase<ObjectT>>;

    namespace impl
    {
      template <typename ObjectT>
      struct ObjectWrapperTraitsBaseImpl
      {
        template <typename T>
        static auto check(T) -> std::bool_constant<std::is_base_of_v<ObjectT, typename T::type>>;
        static auto check(...) -> std::false_type;
      };
    }

    template <typename ObjectT,typename T>
    using  ObjectWrapperTraitsBase = decltype(impl::ObjectWrapperTraitsBaseImpl<ObjectT>::check(std::declval<T>()));
    template <typename ObjectT,typename T>
    struct ObjectWrapperArrayTraitsBase : std::false_type {};
    template <typename ObjectT,typename T>
    struct ObjectWrapperArrayTraitsBase<ObjectT,Array<T>> : std::bool_constant<ObjectWrapperTraitsBase<ObjectT,T>::value> {};

    template <typename ObjectWrapperT>
    struct ObjectWrapperUtilsBase {
      using object_type = typename ObjectWrapperT::type;
      template<typename ObjectT, std::enable_if_t<std::is_base_of_v<ObjectT, object_type>,nullptr_t> = nullptr>
      static auto toObject(const ObjectWrapperT& v) noexcept -> std::shared_ptr<ObjectT> {
        if (v.isConvertible(ObjectT::TypeString())) {
          return std::static_pointer_cast<ObjectT>(v.getObject());
        }
        else {
          return nullptr;
        }
      }
      template<typename ObjectT, std::enable_if_t<std::is_base_of_v<ObjectT, object_type>, nullptr_t> = nullptr>
      static auto toObjects(const std::vector<ObjectWrapperT>& v) noexcept -> std::vector<std::shared_ptr<ObjectT>> {
        std::vector<std::shared_ptr<ObjectT>> res = {};
        for (auto& elm : v) {

          if (elm.isConvertible(ObjectT::TypeString())) {
            res.push_back(std::static_pointer_cast<ObjectT>(elm.getObject()));
          }
        }
        return res;
      }

      template<typename ObjectT, std::enable_if_t<std::is_base_of_v<ObjectT, object_type>, nullptr_t> = nullptr>
      static auto toObjectWrapper(const std::shared_ptr<ObjectT>& v) noexcept -> ObjectWrapperT {
        if (v->isConvertible(ObjectWrapperT::type::TypeString())) {
          return ObjectWrapperT(std::static_pointer_cast<ObjectWrapperT::type>(v));
        }
        else {
          return ObjectWrapperT();
        }
      }
      template<typename ObjectT, std::enable_if_t<std::is_base_of_v<ObjectT, object_type>, nullptr_t> = nullptr>
      static auto toObjectWrappers(const std::vector<std::shared_ptr<ObjectT>>& v) noexcept -> std::vector<ObjectWrapperT> {
        std::vector<ObjectWrapperT> res = {};
        for (auto& elm : v) {
          if (elm->isConvertible(ObjectWrapperT::type::TypeString())) {
            res.push_back(ObjectWrapperT(std::static_pointer_cast<ObjectWrapperT::type>(elm)));
          }
        }
        return res;
      }


    };

    template<typename ObjectT>
    struct PropertyCommonDefinitionsBase {
      using data_types       = PropertyDataTypes;
      using array_data_types = PropertyArrayDataTypes;
      using types            = PropertyTypesBase<ObjectT>;
      static inline constexpr size_t kTypeSwitchIdxDataTypes          = 0;
      static inline constexpr size_t kTypeSwitchIdxArrayDataTypes     = 1;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapper      = 2;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapperArray = 3;
      static inline constexpr size_t kTypeSwitchIdxDefault            = 4;
    };

    template<typename ObjectT, typename T>
    struct PropertyTraitsBase : public PropertyCommonDefinitionsBase<ObjectT>{
      using data_types       = PropertyDataTypes;
      using array_data_types = PropertyArrayDataTypes;
      using types            = PropertyTypesBase<ObjectT>;
      using traits_data_types       = in_tuple<T, data_types>;
      using traits_array_data_types = in_tuple<T, array_data_types>;
      using traits_object_wrapper   = ObjectWrapperTraitsBase<ObjectT, T>;
      using traits_object_wrapper_array = ObjectWrapperArrayTraitsBase<ObjectT, T>;
      static inline constexpr size_t kTypeSwitchIdx = find_integer_sequence<bool,true,std::integer_sequence<bool,
        traits_data_types::value,
        traits_array_data_types::value,
        traits_object_wrapper::value,
        traits_object_wrapper_array::value
      >>::value;
      static inline constexpr bool value = kTypeSwitchIdx < kTypeSwitchIdxDefault;
    };

    template<typename ObjectT>
    struct PropertyBase {
      using data_types                                                = typename PropertyCommonDefinitionsBase<ObjectT>::data_types;
      using array_data_types                                          = typename PropertyCommonDefinitionsBase<ObjectT>::array_data_types;
      using types                                                     = typename PropertyCommonDefinitionsBase<ObjectT>::types;
      template<typename T> using traits_data_types                    = typename PropertyTraitsBase<ObjectT, T>::traits_data_types;
      template<typename T> using traits_array_data_types              = typename PropertyTraitsBase<ObjectT, T>::traits_array_data_types;
      template<typename T> using traits_object_wrapper                = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper;
      template<typename T> using traits_object_wrapper_array          = typename PropertyTraitsBase<ObjectT, T>::traits_object_wrapper_array;
      static inline constexpr size_t kTypeSwitchIdxDataTypes          = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDataTypes;
      static inline constexpr size_t kTypeSwitchIdxArrayDataTypes     = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxArrayDataTypes;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapper      = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapper;
      static inline constexpr size_t kTypeSwitchIdxObjectWrapperArray = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxObjectWrapperArray;
      static inline constexpr size_t kTypeSwitchIdxDefault            = PropertyCommonDefinitionsBase<ObjectT>::kTypeSwitchIdxDefault;
      using variant_type = typename variant_from_tuple<types>::type;
    private:
      template<typename T, size_t idx = PropertyTraitsBase<ObjectT, T>::kTypeSwitchIdx>
      struct impl_type_switch : std::false_type {};
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxDataTypes> : std::true_type {
        using result_type = Option<T>;
        static void setValue(variant_type& res, const T& v) noexcept { res = variant_type(v); }
        static auto getValue(const variant_type& v) noexcept ->result_type {
          auto p = std::get_if<T>(&v);
          if (!p) { return result_type(); } else { return result_type(*p); }
        }
        static bool hasValue(const variant_type& v) noexcept {
          return std::get_if<T>(&v) != nullptr;
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxArrayDataTypes> : std::true_type {
        using result_type = T;
        static void setValue(variant_type& res, const T& v) noexcept { res = v; }
        static auto getValue(const variant_type& v) noexcept ->result_type {
          auto p = std::get_if<T>(&v);
          if (!p) { return result_type(); } else { return result_type(*p); }
        }
        static bool hasValue(const variant_type& v) noexcept {
          return std::get_if<T>(&v) != nullptr;
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxObjectWrapper> : std::true_type {
        using result_type = T;
        static void setValue(variant_type& res, const T& v) noexcept {
          auto tmp = ObjectWrapperUtilsBase<T>::toObject<ObjectT>(v);
          if (!tmp) { res = {}; } else { res = tmp; }
        }
        static auto getValue(const variant_type& v) noexcept ->result_type {
          auto p = std::get_if<std::shared_ptr<ObjectT>>(&v);
          if (!p) { return result_type(); }
          else { return result_type(ObjectWrapperUtilsBase<T>::toObjectWrapper<ObjectT>(*p)); }
        }
        static bool hasValue(const variant_type& v) noexcept {
          auto p = std::get_if<std::shared_ptr<ObjectT>>(&v);
          if (!p) { return false; }
          if (!(*p)->isConvertible(T::type::TypeString)) { return false; }
          return true;
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxObjectWrapperArray> : std::true_type {
        using result_type = T;
        static void setValue(variant_type& res, const T& v) noexcept {
          auto tmp = ObjectWrapperUtilsBase<std::remove_reference_t<decltype(std::declval<T&>()[0])>>::toObjects<ObjectT>(v);
          res = tmp;
        }
        static auto getValue(const variant_type& v) noexcept ->result_type {
          auto p = std::get_if<std::vector<std::shared_ptr<ObjectT>>>(&v);
          if (!p) { return result_type(); }
          else { return result_type(ObjectWrapperUtilsBase<
            std::remove_reference_t<decltype(std::declval<T&>()[0])>
          >::toObjectWrappers<ObjectT>(*p)); }
        }
        static bool hasValue(const variant_type& v) noexcept {
          auto p = std::get_if<std::vector<std::shared_ptr<ObjectT>>> (&v);
          for (auto& elm : *p) {
            if (elm->isConvertible(std::remove_reference_t<decltype(std::declval<T&>()[0])>::type::TypeString)) { return true; }
          }
          return false;
        }
      };
    public:
      PropertyBase() noexcept : m_data{std::monostate()} {}
      PropertyBase(nullptr_t) noexcept : m_data{ std::monostate() } {}
      PropertyBase(const PropertyBase&) noexcept = default;
      PropertyBase(PropertyBase&&) noexcept = default;

      PropertyBase& operator=(nullptr_t) noexcept { m_data = std::monostate(); return *this; }
      PropertyBase& operator=(const PropertyBase&) noexcept = default;
      PropertyBase& operator=(PropertyBase&&) noexcept = default;

      PropertyBase(const std::shared_ptr<ObjectT>& object) noexcept :m_data{ object } {}
      PropertyBase& operator=(const std::shared_ptr<ObjectT>& v) noexcept
      {
        m_data = v;
        return *this;
      }
      PropertyBase(const Array<std::shared_ptr<ObjectT>>& objects) noexcept :m_data{ objects } {}
      PropertyBase& operator=(const Array<std::shared_ptr<ObjectT>>& v) noexcept
      {
        m_data = v;
        return *this;
      }

      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      explicit PropertyBase(const T& v) noexcept : m_data{} { impl_type_switch<T>::setValue(m_data,v); }
      template <typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      PropertyBase& operator=(const T& v) noexcept
      {
        impl_type_switch<T>::setValue(m_data, v);
        return *this;
      }

      template <typename T, std::enable_if_t<in_tuple<T, data_types>::value, nullptr_t> = nullptr>
      explicit PropertyBase(const Option<T>& v) noexcept : m_data{} { if (v) { m_data = *v; } }
      template <typename T, std::enable_if_t<in_tuple<T, data_types>::value, nullptr_t> = nullptr>
      PropertyBase& operator=(const Option<T>& v) noexcept
      {
        if (v)
        {
          m_data = *v;
        }
        else
        {
          m_data = {};
        }
        return *this;
      }

      template <size_t N> explicit PropertyBase(const char(&name)[N]) noexcept : m_data{ Str(name) } {}
      template <size_t N> PropertyBase& operator=(const char(&name)[N]) noexcept
      {
        m_data = Str(name);
        return *this;
      }

      explicit operator Bool() const { return m_data.index() != tuple_size<types>::value; }

      auto getTypeIndex() const -> size_t { return m_data.index(); }
      Bool getTypeIndex(size_t& type_index) const
      {
        auto idx = m_data.index();
        if (idx == PropertyTypeIndexBase<ObjectT, std::monostate>::value) {
          return false;
        }
        type_index = idx;
        return true;
      }
      auto getTypeString() const noexcept -> Str {
        // 基本的にはTYPEに対応するType2Stringを取得
        return std::visit([](const auto& v) {
          using type = std::remove_cv_t < std::remove_reference_t<decltype(v)>>;
          if constexpr (std::is_same_v<type, std::vector<std::shared_ptr<ObjectT>>>) {
            return Str("Array<" + Str(ObjectT::TypeString) + ">");
          }
          else if constexpr (std::is_same_v<type, std::shared_ptr<ObjectT>>) {
            return Str(v->getTypeString());
          }
          else if constexpr (std::is_same_v<type, std::monostate>) {
            return Str("None");
          }
          else {
            return Str(Type2String<type>::value);
          }
        }, m_data);
      }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      Bool hasValue() const noexcept  { return impl_type_switch<T>::hasValue(m_data); }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      auto getValue() const noexcept -> typename impl_type_switch<T>::result_type { return impl_type_switch<T>::getValue(m_data); }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      void setValue(const T& value) noexcept { return impl_type_switch<T>::setValue(m_data,value); }

      auto toVariant() const noexcept-> variant_type { return m_data; }

      template<typename T>
      friend class PropertyBlockBase;
    private:
      variant_type m_data = std::monostate();
    };

    template<typename ObjectT>
    struct PropertyBlockBase
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
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxDataTypes> : std::true_type {
        using result_type = Option<T>;
        static void setValue(PropertyBlockBase<ObjectT>& pb, const Str& name, const T& v) noexcept {
          pb.impl_setValue(name,v);
        }
        static auto getValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) noexcept -> result_type {
          T res = {};
          if (pb.impl_getValue<T>(name,res)) { return res; }
          return std::nullopt;
        }
        static Bool hasValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) {
          auto iter = pb.m_types.find(name);
          if (iter == pb.m_types.end()) { return false; }
          return iter->second.first == PropertyTypeIndexBase<ObjectT, T>::value;
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxArrayDataTypes> : std::true_type {
        using result_type = T;
        static void setValue(PropertyBlockBase<ObjectT>& pb, const Str& name, const T& v) noexcept { pb.impl_setValue(name, v); }
        static auto getValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) noexcept -> result_type {
          T res = {};
          if (pb.impl_getValue<T>(name, res)) { return res; }
          return {};
        }
        static Bool hasValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) {
          auto iter = pb.m_types.find(name);
          if (iter == pb.m_types.end()) { return false; }
          return iter->second.first == PropertyTypeIndexBase<ObjectT, T>::value;
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxObjectWrapper> : std::true_type {
        using result_type = T;
        static void setValue(PropertyBlockBase<ObjectT>& pb, const Str& name, const T& v) noexcept {
          auto tmp = ObjectWrapperUtilsBase<T>::toObject<ObjectT>(v);
          pb.impl_setValue(name, tmp);
        }
        static auto getValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) noexcept -> result_type {
          std::shared_ptr<ObjectT> res = {};
          if (pb.impl_getValue(name, res)) {
            return ObjectWrapperUtilsBase<T>::toObjectWrapper<ObjectT>(res);
          }
          return {};
        }
        static Bool hasValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) {
          auto iter = pb.m_types.find(name);
          if (iter == pb.m_types.end()) { return false; }
          if (iter->second.first != PropertyTypeIndexBase < ObjectT, std::shared_ptr<ObjectT>> ::value) { return false; }
          auto& data = std::get<Storage<std::shared_ptr<ObjectT>>>(m_datas)[iter->second.second];
          return data->isConvertible(T::type::TypeString);
        }
      };
      template<typename T>
      struct impl_type_switch<T, kTypeSwitchIdxObjectWrapperArray> : std::true_type {
        using result_type = T;
        static void setValue(PropertyBlockBase<ObjectT>& pb, const Str& name, const T& v) noexcept {
          auto tmp = ObjectWrapperUtilsBase<std::remove_reference_t<decltype(std::declval<T&>()[0])>>::toObjects<ObjectT>(v);
          pb.impl_setValue(name, tmp);
        }
        static auto getValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) noexcept -> result_type {
          std::vector<std::shared_ptr<ObjectT>> res = {};
          if (pb.impl_getValue(name, res)) {
            return ObjectWrapperUtilsBase<std::remove_reference_t<decltype(std::declval<T&>()[0])>>::toObjectWrappers<ObjectT>(res);
          }
          return {};
        }
        static Bool hasValue(const PropertyBlockBase<ObjectT>& pb, const Str& name) {
          auto iter = pb.m_types.find(name);
          if (iter == pb.m_types.end()) { return false; }
          if (iter->second.first != PropertyTypeIndexBase < ObjectT, std::vector<std::shared_ptr<ObjectT>>> ::value) { return false; }
          auto& data = std::get<Storage<std::vector<std::shared_ptr<ObjectT>>>>(pb.m_datas)[iter->second.second];
          for (auto& elm : data) {
            if (elm->isConvertible(typename std::remove_reference_t<decltype(std::declval<T&>()[0])>::type::TypeString)) { return true; }
          }
          return false;
        }
      };
    private:
      template <typename T>
      struct Storage
      {
        using type = T;
        Storage() noexcept = default;
        Storage(const Storage&) noexcept = default;
        Storage& operator=(const Storage&) noexcept = default;
        //auto operator[](size_t idx) const -> const T& { return m_data[idx]; }
        //auto operator[](size_t idx) -> T& { return m_data[idx]; }
        auto getValue(size_t idx) const -> T { return m_data.at(idx); }
        //void setValue(size_t idx, const T& val) { m_datas[idx] = val; }

        auto insert(T value) -> size_t
        {
          auto old_size = m_data.size();
          if (m_last_empty_location < old_size)
          {
            size_t res = m_last_empty_location;
            m_data[m_last_empty_location] = value;
            auto new_empty_location = m_next_empty_locations[m_last_empty_location];
            for (size_t i = 0; i <= m_last_empty_location; ++i)
            {
              m_next_empty_locations[i] = new_empty_location;
            }
            m_last_empty_location = new_empty_location;
            return res;
          }
          else
          {
            m_data.push_back(value);
            m_next_empty_locations.push_back(0);
            m_last_empty_location = m_data.size();
            return old_size;
          }
        }
        void erase(size_t idx)
        {
          if (m_last_empty_location == idx)
          {
            return;
          }
          if (m_last_empty_location > idx)
          {
            for (size_t i = 0; i < idx; ++i)
            {
              m_next_empty_locations[i] = idx;
            }
            for (size_t i = idx; i < m_last_empty_location; ++i)
            {
              m_next_empty_locations[i] = m_last_empty_location;
            }
            m_data[idx] = {};
            m_last_empty_location = idx;
            return;
          }
          auto idx_m_1 = idx - 1;
          if (m_next_empty_locations[idx_m_1] == idx)
          {
            return;
          }
          auto first = std::distance(
            std::find(
              std::begin(m_next_empty_locations) + m_last_empty_location,
              std::begin(m_next_empty_locations) + idx,
              m_next_empty_locations[idx]),
            std::begin(m_next_empty_locations));
          {
            for (size_t i = first; i < idx; ++i)
            {
              m_next_empty_locations[i] = idx;
            }
            m_data[idx] = {};
          }
        }
      private:
        std::vector<T> m_data = {};
        U16 m_last_empty_location = 0;
        std::vector<U16> m_next_empty_locations = {};
      };
    public:
      struct PropertyRef
      {
        PropertyRef(const PropertyRef&) noexcept = delete;
        PropertyRef(PropertyRef&&) noexcept = delete;
        PropertyRef& operator=(const PropertyRef&) = delete;
        PropertyRef& operator=(PropertyRef&&) = delete;

        void operator=(nullptr_t) noexcept {
          if (m_holder) { m_holder->popValue(m_name); }
        }
        void operator=(const PropertyBase<ObjectT>& prop) noexcept
        {
          if (m_holder) { m_holder->setValue(m_name, prop) };
        }
        template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
        void operator=(const T& value) noexcept
        {
          if (m_holder) { m_holder->setValue(m_name, value); };
        }
        template <size_t N>
        void operator=(const char(&value)[N])
        {
          if (m_holder) { m_holder->setValue(m_name, Str(value)); };
        }
        operator PropertyBase<ObjectT>() const
        {
          if (!m_holder) { return {}; }
          return m_holder->getValue(m_name);
        }

        Bool getTypeIndex(size_t& type_index) const
        {
          if (!m_holder) { return false; }
          return m_holder->getTypeIndex(type_index);
        }
        auto getTypeString() const noexcept -> Str {
          //if (!m_object) { return "None"; }
          //size_t type_idx = 0;
          //auto iter = m_object->m_types.find(m_name);
          //if (iter == m_object->m_types.end()) { return "None"; }
          //// 基本的にはTYPEに対応するType2Stringを取得
          //std::string res = "None";
          //// TODO: 型を取得できるように

          return PropertyBase<ObjectT>(*this).getTypeString();
        }

        template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
        Bool hasValue() const noexcept { if (!m_holder) { return false; };return m_holder->hasValue<T>(m_name); }

        template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
        auto getValue() const noexcept -> typename impl_type_switch<T>::result_type {
          if (!m_holder) { return {}; }
          return m_holder->getValue<T>(m_name);
        }

        template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
        void setValue(const T& value) noexcept {
          if (m_holder) { m_holder->setValue(m_name, value) };
        }
      private:
        friend class PropertyBlockBase;
        PropertyRef(PropertyBlockBase* pb, const Str& name) : m_holder{ pb }, m_name{ name } {}
        PropertyBlockBase* m_holder;
        Str m_name;
      };

      PropertyBlockBase() noexcept : m_datas{} {}
      PropertyBlockBase(const PropertyBlockBase&) noexcept = default;
      PropertyBlockBase(PropertyBlockBase&&) noexcept = default;
      PropertyBlockBase& operator=(const PropertyBlockBase&) noexcept = default;
      PropertyBlockBase& operator=(PropertyBlockBase&&) noexcept = default;

      auto operator[](const Str& name) -> PropertyRef { return PropertyRef(this, name); }
      auto operator[](const Str& name) const -> PropertyBase<ObjectT> { return getValue(name); }

      auto getKeys() const -> std::vector<Str>
      {
        std::vector<Str> res = {};
        for (auto& [key, val] : m_types)
        {
          res.push_back(key);
        }
        return res;
      }
      Bool getTypeIndex(const Str& name, size_t& type_index) const
      {
        auto iter = m_types.find(name);
        if (iter != m_types.end())
        {
          type_index = iter->second.second;
          return true;
        }
        return false;
      }

      bool getValue(const Str& name, PropertyBase<ObjectT>& prop) const
      {
        auto iter = m_types.find(name);
        if (iter != m_types.end())
        {
          prop = impl_getValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<types>::value>());
          return true;
        }
        else
        {
          return false;
        }
      }
      auto getValue(const Str& name) const -> PropertyBase<ObjectT>
      {
        auto iter = m_types.find(name);
        if (iter != m_types.end())
        {
          return impl_getValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<types>::value>());
        }
        else
        {
          return PropertyBase<ObjectT>();
        }
      }
      void setValue(const Str& name, const PropertyBase<ObjectT>& prop)
      {
        popValue(name);
        if (!(!prop))
        {
          std::visit([name, this](const auto& v)
            {
              using v_type = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
              if constexpr (!std::is_same_v<v_type, std::monostate>) {
                m_types.insert({ name, { tuple_index<v_type,types>::value, std::get<Storage<v_type>>(m_datas).insert(v)} });
              }
            },
            prop.m_data);
        }
      }
      Bool hasValue(const Str& name) const { return m_types.count(name) > 0; }

      template<typename T, std::enable_if_t<impl_type_switch<T>::value,nullptr_t> = nullptr>
      auto getValue(const Str& name) const noexcept -> typename impl_type_switch<T>::result_type {
        return impl_type_switch<T>::getValue(*this, name);
      }
      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      void setValue(const Str& name, const T& value) noexcept{
        popValue(name);
        return impl_type_switch<T>::setValue(*this, name, value);
      }
      template<typename T, std::enable_if_t<impl_type_switch<T>::value, nullptr_t> = nullptr>
      Bool hasValue(const Str& name) const { return impl_type_switch<T>::hasValue(*this, name); }

      template <size_t N>
      void setValue(const Str& name, const char(&value)[N]) noexcept { impl_setValue(name, Str(value)); }
      void popValue(const Str& name)
      {
        auto iter = m_types.find(name);
        if (iter != m_types.end())
        {
          impl_popValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<types>::value>());
        }
      }

      void clear()
      {
        m_types = {};
        m_datas = {};
      }

    private:
      template <typename T, std::enable_if_t<in_tuple<T, types>::value, nullptr_t> = nullptr>
      bool impl_getValue(const Str& name, T& value) const noexcept
      {
        auto iter = m_types.find(name);
        if (iter == m_types.end()) { return false; }
        if (iter->second.first != PropertyTypeIndexBase<ObjectT,T>::value) { return false; }
        value = std::get<Storage<T>>(m_datas).getValue(iter->second.second);
        return true;
      }
      template <typename T, std::enable_if_t<in_tuple<T, types>::value, nullptr_t> = nullptr>
      void impl_setValue(const Str& name, const T& value) noexcept
      {
        popValue(name);
        auto val = std::get<Storage<T>>(m_datas).insert(value);
        m_types.insert({ name, {tuple_index<T, types>::value, val } });
      }
      template <size_t... Is>
      auto impl_getValue_for_each(U8 type_idx, size_t data_idx, std::index_sequence<Is...>) const -> PropertyBase<ObjectT>
      {
        PropertyBase<ObjectT> res = {};
        using Swallow = int[];
        (void)Swallow {
          (Is, (type_idx == Is) ? ((res.m_data = std::get<Is>(m_datas).getValue(data_idx)), 0) : 0)...
        };
        if (type_idx == 10) {
          std::cout << std::get<10>(m_datas).getValue(data_idx) << std::endl;
        }
        return res;
      }
      template <size_t... Is>
      void impl_popValue_for_each(U8 type_idx, size_t data_idx, std::index_sequence<Is...>)
      {
        using Swallow = int[];
        (void)Swallow {
          (Is, (type_idx == Is) ? (std::get<Is>(m_datas).erase(data_idx), 0) : 0)...
        };
      }
    private:
      hikari::Dict<Str, Pair<U8, U16>>               m_types = {};
      typename transform_tuple<Storage, types>::type m_datas = {};
    };

    template<typename T, typename ObjectT>
    auto safe_numeric_cast(const PropertyBase<ObjectT>& p) -> decltype(std::enable_if_t<in_tuple<T, PropertyNumericTypes>::value, nullptr_t>{nullptr},Option<T>()) {
      return std::visit([](const auto& p) {
        using arg_type = std::remove_cv_t<std::remove_reference_t<decltype(p)>>;
        if constexpr (in_tuple<arg_type, PropertyNumericTypes>::value) {
          return safe_numeric_cast<T>(p);
        }
        else {
          return Option<T>(std::nullopt);
        }
      }, p.toVariant());
    }
  }
}
