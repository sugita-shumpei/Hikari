#pragma once
#include <type_traits>
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>
#include <nlohmann/json.hpp>
#include <hikari/core/data_type.h>
#include <hikari/core/tuple.h>
#include <hikari/core/transform.h>
namespace hikari
{
    inline namespace core
    {
        struct Object;
        struct ObjectWrapper;

        using PropertyDataTypes    = typename concat_tuple<DataTypes, std::tuple<Transform>>::type;
        using PropertySingleTypes  = typename hikari::concat_tuple<PropertyDataTypes, std::tuple<std::shared_ptr<Object>>>::type;
        using PropertyArrayTypes   = typename hikari::trasform_tuple<std::vector, PropertySingleTypes>::type;
        using PropertyNonNullTypes = PropertyDataTypes;
        using PropertyTypes        = hikari::concat_tuple<PropertySingleTypes,PropertyArrayTypes>::type;

        template <typename T>
        using PropertyTypeIndex = tuple_index<T, PropertyTypes>;

        template <typename T>
        using PropertyTraits = in_tuple<T, PropertyTypes>;
        namespace impl
        {
            struct ObjectWrapperTraitsImpl
            {
                template <typename T>
                static auto check(T) -> std::bool_constant<std::is_base_of_v<Object, typename T::ObjectType>>;
                static auto check(...) -> std::false_type;
            };
        }

        template <typename T>
        using  ObjectWrapperTraits = decltype(impl::ObjectWrapperTraitsImpl::check(std::declval<T>()));
        template <typename T>
        struct ObjectWrapperArrayTraits : std::false_type{};
        template <typename T>
        struct ObjectWrapperArrayTraits<std::vector<T>> : std::bool_constant<ObjectWrapperTraits<T>::value>{};


        // Property:
        // プロパティとして使用可能な型の共用体クラス
        //
        struct Property
        {
        public:
            template <typename T>
            using Traits = in_tuple<T, PropertyTypes>;

            template <typename T>
            using TypeIndex = PropertyTypeIndex<T>;

            Property() noexcept : m_data{} {}
            Property(const Property &) noexcept = default;
            Property(Property &&) noexcept = default;
            Property &operator=(const Property &) noexcept = default;
            Property &operator=(Property &&) noexcept = default;

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            explicit Property(T v) noexcept : m_data{v} {}

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            Property &operator=(T v) noexcept
            {
                m_data = v;
                return *this;
            }

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            Property &operator=(const Option<T> &v) noexcept
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

            template <size_t N>
            explicit Property(const char (&name)[N]) noexcept : m_data{Str(name)} {}
            template <size_t N>
            Property &operator=(const char (&name)[N]) noexcept
            {
                m_data = Str(name);
                return *this;
            }

            explicit operator Bool() const { return m_data.index() != tuple_size<PropertyTypes>::value; }

            auto getString() const -> Str;
            auto getJSONString() const -> Str;

            auto getTypeIndex() const -> size_t { return m_data.index(); }
            auto getInteger() const -> Option<I64>;
            auto getFloat() const -> Option<F64>;
            auto getVector() const -> Option<Vec4>;
            auto getMatrix() const -> Option<Mat4>;

            auto getIntegers() const -> std::vector<I64>;
            auto getFloats() const -> std::vector<F64>;
            auto getVectors() const -> std::vector<Vec4>;
            auto getMatrices() const -> std::vector<Mat4>;
        private:
            inline static constexpr size_t kTypeSwitchIndexProperty           = 0;
            inline static constexpr size_t kTypeSwitchIndexObjectWrapper      = 1;
            inline static constexpr size_t kTypeSwitchIndexObjectWrapperArray = 2;
            template <typename T, size_t N> struct impl_type_switch_2;
            template <typename T> struct impl_type_switch_2<T, kTypeSwitchIndexProperty>
            {
                using ResultType = std::conditional_t<in_tuple<T, PropertyNonNullTypes>::value, Option<T>, T>;
                static Bool getValue_1(const Property &p, T &value) noexcept
                {
                    auto p_val = std::get_if<T>(&p.m_data);
                    if (p_val)
                    {
                        value = *p_val;
                        return true;
                    }
                    return false;
                }
                static auto getValue_2(const Property &p) noexcept -> ResultType
                {
                    ResultType res;
                    auto p_val = std::get_if<T>(&p.m_data);
                    if (p_val)
                    {
                        res = *p_val;
                    }
                    return res;
                }
                static void setValue_1(const Property &p, T value) noexcept { p.m_data = value; }
            };
            template <typename T> struct impl_type_switch_2<T, kTypeSwitchIndexObjectWrapper>
            {
                using ResultType = T;
                static Bool getValue_1(const Property &p, T &value) noexcept
                {
                    std::shared_ptr<Object> ptr;
                    if (!impl_type_switch_2<decltype(ptr), 0>::getValue_1(p, ptr))
                    {
                        obj = nullptr;
                        return false;
                    }
                    if (!obj->isConvertible(ObjectWrapperLike::ObjectType::TypeString()))
                    {
                        return false;
                    }
                    obj = std::static_pointer_cast<typename ObjectWrapperLike::ObjectType>(ptr);
                    return true;
                }
                static auto getValue_2(const Property &p) noexcept -> ResultType
                {
                    using ObjectWrapperLike = T;
                    T res;
                    if (getValue_1(p, res))
                    {
                        return res;
                    }
                    return T();
                }
                static void setValue_1(const Property &p, const T &obj) noexcept { p.m_data = std::static_pointer_cast<Object>(obj.getObject()); }
            };
            template <typename T> struct impl_type_switch_2<T, kTypeSwitchIndexObjectWrapperArray>
            {
                using ResultType = T;
                static Bool getValue_1(const Property &p, T &value) noexcept
                {
                    using ObjectWrapperLike = std::remove_reference_t<decltype(std::declval<T>()[0])>;
                    value = {};
                    std::vector<std::shared_ptr<Object>> arr;
                    if (!impl_type_switch_2<decltype(arr), 0>::getValue_1(p, arr))
                    {
                        value = {};
                        return false;
                    }
                    value.reserve(arr.size());
                    for (auto &elm : arr)
                    {
                        if (!elm)
                        {
                            value.push_back(ObjectWrapperLike(nullptr));
                        }
                        else if (elm->isConvertible(ObjectWrapperLike::ObjectType::TypeString()))
                        {
                            value.push_back(ObjectWrapperLike(std::static_pointer_cast<typename ObjectWrapperLike::ObjectType>(elm)));
                        }
                        else
                        {
                            value.clear();
                            return false;
                        }
                    }
                    return true;
                }
                static auto getValue_2(const Property &p) noexcept -> ResultType
                {
                    T res;
                    if (getValue_1(p, res))
                    {
                        return res;
                    }
                    return T();
                }
                static void setValue_1(const Property &p, const T &obj) noexcept
                {
                    using ObjectWrapperLike = std::remove_reference_t<decltype(std::declval<T>()[0])>;
                    auto data = std::vector<std::shared_ptr<Object>>();
                    data.reserve(objects.size());
                    std::copy(objects.begin(), objects.end(), std::back_inserter(data), [](const ObjectWrapperLike &v)
                              { return v.getObject(); });
                    m_data = data;
                }
            };
            template <typename T> using TypeSwitchPatternSequence = std::integer_sequence<bool, PropertyTraits<T>::value, ObjectWrapperTraits<T>::value, ObjectWrapperArrayTraits<T>::value>;
        public:
            template <typename T, size_t idx = find_integer_sequence<bool, true, TypeSwitchPatternSequence<T>>::value>
            Bool getValue(T &value) const noexcept { return impl_type_switch_2<T, idx>::getValue_1(*this, value); }
            template <typename T, size_t idx = find_integer_sequence<bool, true, TypeSwitchPatternSequence<T>>::value>
            auto getValue() const noexcept -> typename impl_type_switch_2<T, idx>::ResultType { return impl_type_switch_2<T, idx>::getValue_2(*this); }
            template <typename T, size_t idx = find_integer_sequence<bool, true, TypeSwitchPatternSequence<T>>::value>
            void setValue(const T &value) noexcept { impl_type_switch_2<T, idx>::setValue_1(*this, value); }
        private:
            friend struct PropertyBlock;
            friend struct ObjectWrapper;
            typename variant_from_tuple<PropertyTypes>::type m_data = {};
        };
        // PropertyBlock:
        // プロパティと変数をまとめた辞書クラス
        //
        struct PropertyBlock
        {
            template <typename T>
            struct Storage
            {
                Storage() noexcept = default;
                Storage(const Storage &) noexcept = default;
                Storage &operator=(const Storage &) noexcept = default;
                auto operator[](size_t idx) const -> const T & { return m_data[idx]; }
                auto operator[](size_t idx) -> T & { return m_data[idx]; }
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
            using TypesTuple = PropertyTypes;
            using DatasTuple = trasform_tuple<Storage, PropertyTypes>::type;

        public:
            template <typename T>
            using Traits = in_tuple<T, PropertyTypes>;
            struct PropertyRef
            {
                template <typename T>
                using Traits = in_tuple<T, PropertyTypes>;
                PropertyRef(const PropertyRef &) noexcept = delete;
                PropertyRef(PropertyRef &&) noexcept = delete;
                PropertyRef &operator=(const PropertyRef &) = delete;
                PropertyRef &operator=(PropertyRef &&) = delete;
                void operator=(const Property &prop) noexcept
                {
                    m_object->setValue(m_name, prop);
                }
                template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
                void operator=(T value) noexcept
                {
                    m_object->setValue(m_name, value);
                }
                template <size_t N>
                void operator=(const char (&value)[N])
                {
                    m_object->setValue(m_name, Str(value));
                }
                operator Property() const
                {
                    return m_object->getValue(m_name);
                }

            private:
                friend struct PropertyBlock;
                PropertyRef(PropertyBlock *pb, const Str &name) : m_object{pb}, m_name{name} {}
                PropertyBlock *m_object;
                Str m_name;
            };

            PropertyBlock() noexcept : m_datas{} {}
            PropertyBlock(const PropertyBlock &) noexcept = default;
            PropertyBlock(PropertyBlock &&) noexcept = default;
            PropertyBlock &operator=(const PropertyBlock &) noexcept = default;
            PropertyBlock &operator=(PropertyBlock &&) noexcept = default;

            auto operator[](const Str &name) -> PropertyRef { return PropertyRef(this, name); }
            auto operator[](const Str &name) const -> Property { return getValue(name); }

            auto getKeys() const -> std::vector<Str>
            {
                std::vector<Str> res = {};
                for (auto &[key, val] : m_types)
                {
                    res.push_back(key);
                }
                return res;
            }
            Bool getTypeIndex(const Str &name, size_t &type_index) const
            {
                auto iter = m_types.find(name);
                if (iter != m_types.end())
                {
                    type_index = iter->second.second;
                    return true;
                }
                return false;
            }

            void setValue(const Str &name, const Property &prop)
            {
                popValue(name);
                if (!(!prop))
                {
                    std::visit([name, this](const auto &v)
                               {
            using v_type = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
            if constexpr (!std::is_same_v<v_type, std::monostate>) {
              m_types.insert({ name, { tuple_index<v_type,TypesTuple>::value, std::get<Storage<v_type>>(m_datas).insert(v)} });
            } },
                               prop.m_data);
                }
            }
            bool getValue(const Str &name, Property &prop) const
            {
                auto iter = m_types.find(name);
                if (iter != m_types.end())
                {
                    prop = impl_getValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<TypesTuple>::value>());
                    return true;
                }
                else
                {
                    return false;
                }
            }
            auto getValue(const Str &name) const -> Property
            {
                auto iter = m_types.find(name);
                if (iter != m_types.end())
                {
                    return impl_getValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<TypesTuple>::value>());
                }
                else
                {
                    return Property();
                }
            }
            Bool hasValue(const Str &name) const { return m_types.count(name) > 0; }

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            void setValue(const Str &name, T value) noexcept
            {
                popValue(name);
                m_types.insert({name, {tuple_index<T, TypesTuple>::value, std::get<Storage<T>>(m_datas).insert(value)}});
            }
            template <size_t N>
            void setValue(const Str &name, const char (&value)[N]) noexcept { setValue(name, Str(value)); }
            void popValue(const Str &name)
            {
                auto iter = m_types.find(name);
                if (iter != m_types.end())
                {
                    impl_popValue_for_each(iter->second.first, iter->second.second, std::make_index_sequence<tuple_size<TypesTuple>::value>());
                }
            }

            void clear()
            {
                m_types = {};
                m_datas = {};
            }

        private:
            template <size_t... Is>
            auto impl_getValue_for_each(U8 type_idx, size_t data_idx, std::index_sequence<Is...>) const -> Property
            {
                Property res = {};
                using Swallow = int[];
                (void)Swallow{
                    (Is, (type_idx == Is) ? (res = std::get<Is>(m_datas)[data_idx], 0) : 0)...};
                return res;
            }
            template <size_t... Is>
            void impl_popValue_for_each(U8 type_idx, size_t data_idx, std::index_sequence<Is...>)
            {
                using Swallow = int[];
                (void)Swallow{
                    (Is, (type_idx == Is) ? (std::get<Is>(m_datas).erase(data_idx), 0) : 0)...};
            }

        private:
            std::unordered_map<Str, std::pair<U8, U16>> m_types = {};
            DatasTuple m_datas = {};
        };
        // Object:
        // オブジェクト
        //
        struct Object
        {
            static inline Bool Convertible(const Str &str) noexcept
            {
                if (str == TypeString())
                {
                    return true;
                }
                return false;
            }
            static inline constexpr auto TypeString() -> const char * { return "Object"; }
            virtual ~Object() noexcept {}
            virtual auto getTypeString() const -> Str = 0;
            virtual auto getJSONString() const -> Str = 0;
            virtual auto getName() const -> Str = 0;
            virtual auto getPropertyNames() const -> std::vector<Str> = 0;
            virtual void setPropertyBlock(const PropertyBlock &pb) = 0;
            virtual void getPropertyBlock(PropertyBlock &pb) const = 0;
            virtual Bool hasProperty(const Str &name) const = 0;
            virtual Bool setProperty(const Str &name, const Property &value) = 0;
            virtual Bool getProperty(const Str &name, Property &value) const = 0;
            virtual Bool isConvertible(const Str &type_name) const = 0;
            auto getProperty(const Str &name) const -> Property
            {
                Property res;
                getProperty(name, res);
                return res;
            }
        };
        // ObjectPropertyBlock:
        // オブジェクトをPropertyBlockとして扱うためのラッパー
        //
        struct ObjectPropertyRef
        {
        public:
            template <typename T>
            using Traits = in_tuple<T, PropertyTypes>;

            ObjectPropertyRef(const std::shared_ptr<Object> &object, const Str &name) : m_object{object}, m_name{name}
            {
            }

            ObjectPropertyRef(const ObjectPropertyRef &) noexcept = delete;
            ObjectPropertyRef(ObjectPropertyRef &&) noexcept = delete;
            ObjectPropertyRef &operator=(const ObjectPropertyRef &) = delete;
            ObjectPropertyRef &operator=(ObjectPropertyRef &&) = delete;
            void operator=(const Property &prop) noexcept
            {
                auto pb = m_object.lock();
                if (!pb)
                {
                    return;
                }
                pb->setProperty(m_name, prop);
            }

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            void operator=(T value) noexcept
            {
                auto pb = m_object.lock();
                pb->setProperty(m_name, Property(value));
            }
            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            void operator=(const Option<T> &value) noexcept
            {
                auto pb = m_object.lock();
                if (!value)
                {
                    pb->setProperty(m_name, Property());
                }
                else
                {
                    pb->setProperty(m_name, Property(*value));
                }
            }
            template <size_t N>
            void operator=(const char (&value)[N])
            {
                auto pb = m_object.lock();
                pb->setProperty(m_name, Property(Str(value)));
            }

            operator Property() const
            {
                auto pb = m_object.lock();
                if (!pb)
                {
                    return Property();
                }
                Property res;
                if (!pb->getProperty(m_name, res))
                {
                    return Property();
                }
                return res;
            }
            explicit operator Bool() const
            {
                auto pb = m_object.lock();
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

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            auto getValue() const -> std::conditional_t<in_tuple<T, PropertyNonNullTypes>::value, std::optional<T>, T>
            {
                auto p = Property(*this);
                if (!p)
                {
                    return std::conditional_t<in_tuple<T, PropertyNonNullTypes>::value, std::optional<T>, T>();
                }
                return p.getValue<T>();
            }
        private:
            std::weak_ptr<Object> m_object;
            Str m_name;
        };
        // ObjectWrapperImpl:
        // オブジェクトポインタをハンドルとして扱うための基底クラス
        //
        struct ObjectRefHolderUtils
        {
            template <typename ObjectDerive, std::enable_if_t<std::is_base_of_v<Object, ObjectDerive>, nullptr_t> = nullptr>
            static auto getRef(const std::shared_ptr<ObjectDerive> &ptr) noexcept -> std::shared_ptr<ObjectDerive> { return ptr; }
            template <typename ObjectDerive, std::enable_if_t<std::is_base_of_v<Object, ObjectDerive>, nullptr_t> = nullptr>
            static auto getRef(const std::weak_ptr<ObjectDerive> &ptr) noexcept -> std::shared_ptr<ObjectDerive> { return ptr.lock(); }
        };
        template <template <typename...> typename ObjectRefHolder, typename ObjectDerive, std::enable_if_t<std::is_base_of_v<Object, ObjectDerive>, nullptr_t> = nullptr>
        struct ObjectWrapperImpl
        {
            template <typename T>
            using Traits = in_tuple<T, PropertyTypes>;
            using PropertyRef = ObjectPropertyRef;
            using ObjectType = ObjectDerive;

            ObjectWrapperImpl() noexcept : m_object{} {}
            ObjectWrapperImpl(nullptr_t) noexcept : m_object{nullptr} {}
            ObjectWrapperImpl(const ObjectWrapperImpl &) = default;
            ObjectWrapperImpl(const std::shared_ptr<ObjectType> &object) : m_object{object} {}

            auto getKeys() const -> std::vector<Str>
            {
                auto object = getObject();
                if (!object)
                {
                    return {};
                }
                return object->getPropertyNames();
            }

            auto operator[](const Str &name) -> PropertyRef
            {
                auto object = getObject();
                return PropertyRef(object, name);
            }
            auto operator[](const Str &name) const -> Property { return getValue(name); }
            template <size_t N>
            auto operator[](const char (&name)[N]) -> PropertyRef { return operator[](Str(name)); }
            template <size_t N>
            auto operator[](const char (&name)[N]) const -> Property { return operator[](Str(name)); }

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

            void getPropertyBlock(PropertyBlock &pb) const
            {
                auto object = getObject();
                if (!object)
                {
                    return;
                }
                object->getPropertyBlock(pb);
            }
            void setPropertyBlock(const PropertyBlock &pb)
            {
                auto object = getObject();
                if (!object)
                {
                    return;
                }
                object->setPropertyBlock(pb);
            }

            Bool setValue(const Str &name, const Property &prop)
            {
                auto object = getObject();
                if (!object)
                {
                    return;
                }
                return object->setProperty(name, prop);
            }
            Bool getValue(const Str &name, Property &prop) const
            {
                auto object = getObject();
                if (!object)
                {
                    return false;
                }
                return object->getProperty(name, prop);
            }
            auto getValue(const Str &name) const -> Property
            {
                auto object = getObject();
                if (!object)
                {
                    return Property();
                }
                return object->getProperty(name);
            }
            Bool hasValue(const Str &name) const
            {
                auto object = getObject();
                if (!object)
                {
                    return false;
                }
                return object->hasProperty(name);
            }

            template <typename T, std::enable_if_t<Traits<T>::value, nullptr_t> = nullptr>
            void setValue(const Str &name, T value) noexcept { return setValue(name, Property(value)); }
            template <size_t N>
            void setValue(const Str &name, const char (&value)[N]) noexcept { setValue(name, Str(value)); }

            auto getObject() const -> std::shared_ptr<ObjectDerive> { return ObjectRefHolderUtils::getRef(m_object); }

        protected:
            void setObject(const std::shared_ptr<ObjectDerive> &object) { m_object = object; }

        private:
            ObjectRefHolder<ObjectDerive> m_object;
        };
        // ObjectWrapper:
        // オブジェクトポインタをハンドルとして扱うための実装クラス
        //
        struct ObjectWrapper : protected ObjectWrapperImpl<std::shared_ptr, Object>
        {
            template <typename T>
            using Traits = ObjectWrapperImpl::Traits<T>;
            using PropertyRef = ObjectWrapperImpl::PropertyRef;
            using ObjectType = ObjectWrapperImpl::ObjectType;

            ObjectWrapper() noexcept : ObjectWrapperImpl() {}
            ObjectWrapper(nullptr_t) noexcept : ObjectWrapperImpl(nullptr) {}
            ObjectWrapper(const std::shared_ptr<Object> &object) noexcept : ObjectWrapperImpl(object) {}
            ObjectWrapper(const ObjectWrapper &opb) noexcept : ObjectWrapperImpl(opb.getObject()) {}
            ObjectWrapper(ObjectWrapper &&opb) noexcept : ObjectWrapperImpl(opb.getObject()) { opb.setObject({}); }

            ObjectWrapper &operator=(const ObjectWrapper &opb) noexcept
            {
                auto old_object = getObject();
                auto new_object = opb.getObject();
                if (old_object != new_object)
                {
                    ObjectWrapperImpl::setObject(new_object);
                }
                return *this;
            }
            ObjectWrapper &operator=(ObjectWrapper &&opb) noexcept
            {
                auto old_object = getObject();
                auto new_object = opb.getObject();
                if (old_object != new_object)
                {
                    ObjectWrapperImpl::setObject(new_object);
                    opb.setObject({});
                }
                return *this;
            }
            ObjectWrapper &operator=(const std::shared_ptr<Object> &obj) noexcept
            {
                auto old_object = getObject();
                auto &new_object = obj;
                if (old_object != new_object)
                {
                    ObjectWrapperImpl::setObject(new_object);
                }
                return *this;
            }

            template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<Object, typename ObjectWrapperLike::ObjectType>, nullptr_t> = nullptr>
            ObjectWrapper(const ObjectWrapperLike &wrapper) noexcept : ObjectWrapperImpl(wrapper.getObject()) {}
            template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<Object, typename ObjectWrapperLike::ObjectType>, nullptr_t> = nullptr>
            ObjectWrapper &operator=(const ObjectWrapperLike &wrapper) noexcept
            {
                auto old_object = getObject();
                auto new_object = wrapper.getObject();
                if (old_object != new_object)
                {
                    ObjectWrapperImpl::setObject(new_object);
                }
                return *this;
            }

            using ObjectWrapperImpl::operator!;
            using ObjectWrapperImpl::operator bool;
            using ObjectWrapperImpl::operator[];
            using ObjectWrapperImpl::getKeys;
            using ObjectWrapperImpl::getObject;
            using ObjectWrapperImpl::getPropertyBlock;
            using ObjectWrapperImpl::getValue;
            using ObjectWrapperImpl::hasValue;
            using ObjectWrapperImpl::setPropertyBlock;
            using ObjectWrapperImpl::setValue;
        };
        struct ObjectWrapperUtils
        {
        private:
          inline static constexpr size_t kTypeSwitchIndexObjectWrapper      = 0;
          inline static constexpr size_t kTypeSwitchIndexObjectWrapperArray = 1;
          template <typename To, typename From, size_t N> struct impl_type_switch_2;
          template <typename To, typename From> struct impl_type_switch_2<To, From, kTypeSwitchIndexObjectWrapper>
          {
            static auto convert(const From& from) -> To {
              auto object = from.getObject();
              if (object->isConvertible<typename To::ObjectType>()) { 
                return To(std::static_pointer_cast<typename To::ObjectType>(object));
              }
              else {
                return To(nullptr);
              }
            }
          };
          template <typename To, typename From> struct impl_type_switch_2<To, From, kTypeSwitchIndexObjectWrapperArray>
          {
            static auto convert(const From& from) -> To {
              To res = {};
              using to_elm_type =std::remove_reference_t<decltype(res[0])>;
              for (auto& elm : from) {
                auto object = elm.getObject();
                if (object->isConvertible<typename to_elm_type::ObjectType>()) {
                  res.push_back(to_elm_type(std::static_pointer_cast<typename to_elm_type::ObjectType>(object)));
                }
              }
              return res;
            }
          };

          template <typename To, typename From> using TypeSwitchPatternSequence = std::integer_sequence<bool,
            ObjectWrapperTraits<To>::value      && ObjectWrapperTraits<From>::value,
            ObjectWrapperArrayTraits<To>::value && ObjectWrapperArrayTraits<From>::value 
          >;
        public:
          template <typename To, typename From, size_t idx = find_integer_sequence<bool,true, TypeSwitchPatternSequence<To,From>>::value>
          static auto convert(const From& from) -> To { return impl_type_switch_2<To, From, idx>::convert(from); }
        };
        // FieldObject:
        // PropertyBlockに階層構造を持たせることを可能にしたもの
        //
        struct ObjectPropertyRef;

        template <typename T, std::enable_if_t<in_tuple<T, PropertyDataTypes>::value && !std::is_same_v<T, Quat>, nullptr_t> = nullptr>
        inline auto convertPropertyTypeToString(T v) -> std::string { return convertToString(v); }
        inline auto convertPropertyTypeToString(Quat v) -> std::string { return convertToString(v); }
        inline auto convertPropertyTypeToString(std::monostate v) -> std::string { return "null"; }
        inline auto convertPropertyTypeToString(std::shared_ptr<Object> v) -> std::string { return v ? v->getJSONString() : "null"; }
        template <typename T, std::enable_if_t<in_tuple<T, PropertyArrayTypes>::value, nullptr_t> = nullptr>
        inline auto convertPropertyTypeToString(const T &v) -> std::string
        {
            std::string res = "[ ";
            for (auto i = 0; i < v.size(); ++i)
            {
                res += convertPropertyTypeToString(v[i]);
                if (i != v.size() - 1)
                {
                    res += ", ";
                }
            }
            res += " ]";
            return res;
        }

        template <typename T, std::enable_if_t<in_tuple<T, PropertyDataTypes>::value && !std::is_same_v<T, Quat>, nullptr_t> = nullptr>
        inline auto convertPropertyTypeToJSONString(T v) -> std::string { return convertToJSONString(v); }
        inline auto convertPropertyTypeToJSONString(Quat v) -> std::string { return convertToJSONString(v); }
        inline auto convertPropertyTypeToJSONString(std::monostate v) -> std::string { return convertToJSONString(v); }
        inline auto convertPropertyTypeToJSONString(std::shared_ptr<Object> v) -> std::string { return v ? v->getJSONString() : "null"; }

        template <typename T, std::enable_if_t<in_tuple<T, PropertyArrayTypes>::value, nullptr_t> = nullptr>
        inline auto convertPropertyTypeToJSONString(const T &v) -> std::string
        {
            std::string res = "[ ";
            for (auto i = 0; i < v.size(); ++i)
            {
                res += convertPropertyTypeToString(v[i]);
                if (i != v.size() - 1)
                {
                    res += ", ";
                }
            }
            res += " ]";
            if constexpr (!std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(v[0])>>, std::shared_ptr<Object>>)
            {
                std::string type_str = Type2String<T>::value;
                res = "{ \"type\" : \"" + type_str + "\", \"value\" : " + res + " }";
            }
            return res;
        }

        template <typename T>
        struct ConvertJSONStringToPropertyTypeTraits : std::false_type
        {
        };

#define HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(TYPE)                               \
    template <>                                                                                   \
    struct ConvertJSONStringToPropertyTypeTraits<TYPE> : std::true_type                           \
    {                                                                                             \
        static auto eval(const Str &v) -> Option<TYPE> { return convertFromJSONString<TYPE>(v); } \
    }
#define HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(TYPE)            \
    template <>                                                                      \
    struct ConvertJSONStringToPropertyTypeTraits<std::vector<TYPE>> : std::true_type \
    {                                                                                \
        static auto eval(const Str &v) -> std::optional<std::vector<TYPE>>           \
        {                                                                            \
            nlohmann::json j = nlohmann::json::parse(v);                             \
            if (!j.is_object())                                                      \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            auto iter_type = j.find("type");                                         \
            if (iter_type == j.end())                                                \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            if (!iter_type.value().is_string())                                      \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            auto &type = iter_type.value();                                          \
            if (type != Type2String<std::vector<TYPE>>::value)                       \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            auto &iter_value = j.find("value");                                      \
            if (iter_value == j.end())                                               \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            if (!iter_value.value().is_array())                                      \
            {                                                                        \
                return std::nullopt;                                                 \
            }                                                                        \
            auto &arr = iter_value.value();                                          \
            std::vector<TYPE> res = {};                                              \
            for (auto &elm : arr)                                                    \
            {                                                                        \
                auto tmp = convertFromString<TYPE>(elm.dump());                      \
                if (!tmp)                                                            \
                {                                                                    \
                    return std::nullopt;                                             \
                }                                                                    \
                res.push_back(*tmp);                                                 \
            }                                                                        \
            return res;                                                              \
        }                                                                            \
    }

        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(I8);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(I16);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(I32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(I64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(U8);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(U16);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(U32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(U64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(F32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(F64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Vec2);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Vec3);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Vec4);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Mat2);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Mat3);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Mat4);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Quat);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Bool);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Str);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_DEFINE(Transform);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(I8);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(I16);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(I32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(I64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(U8);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(U16);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(U32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(U64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(F32);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(F64);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Vec2);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Vec3);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Vec4);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Mat2);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Mat3);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Mat4);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Quat);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Bool);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Str);
        HK_CONVERT_JSON_STRING_TO_PROPERTY_TYPE_TRAITS_ARRAY_DEFINE(Transform);

        template <typename T, std::enable_if_t<in_tuple<T, PropertyTypes>::value, nullptr_t> = nullptr>
        inline auto convertJSONStringToPropertyType(const Str &str) -> Option<T>
        {
            return ConvertJSONStringToPropertyTypeTraits<T>::eval(str);
        }

        auto convertPropertyToString(const Property &prop) -> Str;
        auto convertPropertyToJSONString(const Property &prop) -> Str;
        auto convertJSONStringToProperty(const Str &str) -> Property;

        inline auto convertToString(const Property &v) -> Str
        {
            return convertPropertyToString(v);
        }
        inline auto convertToJSONString(const Property &v) -> Str
        {
            return convertPropertyToJSONString(v);
        }
    }
}
