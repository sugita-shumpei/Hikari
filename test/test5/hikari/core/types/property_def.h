#pragma once
#define HK_PROPERTY_TYPE_DEFINE_TYPE(TYPE)        \
    Property(const TYPE &v) : m_data{v} {}        \
    Property &operator=(const TYPE &v)            \
    {                                             \
        m_data = v;                               \
        return *this;                             \
    }                                             \
    void set##TYPE(const TYPE &v) { m_data = v; } \
    Bool get##TYPE(TYPE &v) const                 \
    {                                             \
        auto p = std::get_if<TYPE>(&m_data);      \
        if (!p)                                   \
        {                                         \
            return false;                         \
        }                                         \
        v = *p;                                   \
        return true;                              \
    }                                             \
    auto get##TYPE() const -> Option<TYPE>        \
    {                                             \
        TYPE v;                                   \
        if (get##TYPE(v))                         \
        {                                         \
            return v;                             \
        }                                         \
        else                                      \
        {                                         \
            return std::nullopt;                  \
        }                                         \
    }

#define HK_PROPERTY_TYPE_DEFINE_VALU(TYPE)                              \
    template <typename T, enabler_t<std::is_same_v<T, TYPE>> = nullptr> \
    Bool setValue(const TYPE &v) { return set##TYPE(v); }               \
    template <typename T, enabler_t<std::is_same_v<T, TYPE>> = nullptr> \
    Bool getValue(TYPE &v) const { return get##TYPE(v); }               \
    template <typename T>                                               \
    auto getValue() const -> std::remove_reference_t<decltype(enabler_t<std::is_same_v<T, TYPE>>{nullptr}, std::declval<Option<T>>())> { return get##TYPE(); }

#define HK_PROPERTY_ARRAY_TYPE_DEFINE_TYPE(TYPE)                \
    Property(const Array##TYPE &v) : m_data{v} {}               \
    Property &operator=(const Array##TYPE &v)                   \
    {                                                           \
        m_data = v;                                             \
        return *this;                                           \
    }                                                           \
    void set##Array##TYPE(const Array##TYPE &v) { m_data = v; } \
    Bool get##Array##TYPE(Array##TYPE &v) const                 \
    {                                                           \
        auto p = std::get_if<Array##TYPE>(&m_data);             \
        if (!p)                                                 \
        {                                                       \
            return false;                                       \
        }                                                       \
        v = *p;                                                 \
        return true;                                            \
    }                                                           \
    auto get##Array##TYPE() const -> Array##TYPE                \
    {                                                           \
        Array##TYPE v;                                          \
        if (get##Array##TYPE(v))                                \
        {                                                       \
            return v;                                           \
        }                                                       \
        else                                                    \
        {                                                       \
            return Array##TYPE();                               \
        }                                                       \
    }

#define HK_PROPERTY_ARRAY_TYPE_DEFINE_VALU(TYPE)                        \
    template <typename T, enabler_t<std::is_same_v<T, TYPE>> = nullptr> \
    Bool setValue(const Array##TYPE &v) { return set##Array##TYPE(v); } \
    template <typename T, enabler_t<std::is_same_v<T, TYPE>> = nullptr> \
    Bool getValue(Array##TYPE &v) const { return get##Array##TYPE(v); } \
    template <typename T>                                               \
    auto getValue() const -> std::remove_reference_t<decltype(enabler_t<std::is_same_v<T, Array##TYPE>>{nullptr}, std::declval<T>())> { return get##Array##TYPE(); }

#define HK_PROPERTY_TYPE_DEFINE(TYPE)                                   \
    HK_PROPERTY_TYPE_DEFINE_TYPE(TYPE);                                 \
    HK_PROPERTY_TYPE_DEFINE_VALU(TYPE);                                 \
    Bool has##TYPE() const { return std::get_if<TYPE>(&m_data); }       \
    template <typename T, enabler_t<std::is_same_v<T, TYPE>> = nullptr> \
    Bool hasValue() const { return has##TYPE(); }

#define HK_PROPERTY_VECTOR_AND_MATRIX_DEFINE()          \
    Bool getVector(Vector &v) const                     \
    {                                                   \
        {                                               \
            Vec2 m;                                     \
            if (getVec2(m))                             \
            {                                           \
                v = Vector(m, 0.0f, 0.0f);              \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            Vec3 m;                                     \
            if (getVec3(m))                             \
            {                                           \
                v = Vector(m, 0.0f);                    \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            Vec4 m;                                     \
            if (getVec4(m))                             \
            {                                           \
                v = Vector(m);                          \
                return true;                            \
            }                                           \
        }                                               \
        return false;                                   \
    }                                                   \
    auto getVector() const -> Option<Vector>            \
    {                                                   \
        Vector v;                                       \
        if (getVector(v))                               \
        {                                               \
            return v;                                   \
        }                                               \
        else                                            \
        {                                               \
            return std::nullopt;                        \
        }                                               \
    }                                                   \
    Bool hasVector() const                              \
    {                                                   \
        if (hasVec2())                                  \
        {                                               \
            return true;                                \
        }                                               \
        if (hasVec3())                                  \
        {                                               \
            return true;                                \
        }                                               \
        if (hasVec4())                                  \
        {                                               \
            return true;                                \
        }                                               \
        return false;                                   \
    }                                                   \
    Bool getMatrix(Matrix &v) const                     \
    {                                                   \
        {                                               \
            Mat2 m;                                     \
            if (getMat2(m))                             \
            {                                           \
                m = Mat4(m);                            \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            Mat3 m;                                     \
            if (getMat3(m))                             \
            {                                           \
                m = Mat4(m);                            \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            Mat4 m;                                     \
            if (getMat4(m))                             \
            {                                           \
                m = Mat4(m);                            \
                return true;                            \
            }                                           \
        }                                               \
        return false;                                   \
    }                                                   \
    auto getMatrix() const -> Option<Matrix>            \
    {                                                   \
        Matrix v;                                       \
        if (getMatrix(v))                               \
        {                                               \
            return v;                                   \
        }                                               \
        else                                            \
        {                                               \
            return std::nullopt;                        \
        }                                               \
    }                                                   \
    Bool hasMatrix() const                              \
    {                                                   \
        if (hasMat2())                                  \
        {                                               \
            return true;                                \
        }                                               \
        if (hasMat3())                                  \
        {                                               \
            return true;                                \
        }                                               \
        if (hasMat4())                                  \
        {                                               \
            return true;                                \
        }                                               \
        return false;                                   \
    }                                                   \
    Bool getArrayVector(ArrayVector &v) const           \
    {                                                   \
        {                                               \
            ArrayVec2 m;                                \
            if (getArrayVec2(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Vector(e, 0.0f, 0.0f)); \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            ArrayVec3 m;                                \
            if (getArrayVec3(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Vector(e, 0.0f));       \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            ArrayVec4 m;                                \
            if (getArrayVec4(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Vector(e));             \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        return false;                                   \
    }                                                   \
    auto getArrayVector() const -> ArrayVector          \
    {                                                   \
        ArrayVector v;                                  \
        if (getArrayVector(v))                          \
        {                                               \
            return v;                                   \
        }                                               \
        else                                            \
        {                                               \
            return ArrayVector();                       \
        }                                               \
    }                                                   \
    Bool hasArrayVector() const                         \
    {                                                   \
        if (hasArrayVec2())                             \
        {                                               \
            return true;                                \
        }                                               \
        if (hasArrayVec3())                             \
        {                                               \
            return true;                                \
        }                                               \
        if (hasArrayVec4())                             \
        {                                               \
            return true;                                \
        }                                               \
        return false;                                   \
    }                                                   \
    Bool getArrayMatrix(ArrayMatrix &v) const           \
    {                                                   \
        {                                               \
            ArrayMat2 m;                                \
            if (getArrayMat2(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Matrix(e));             \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            ArrayMat3 m;                                \
            if (getArrayMat3(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Matrix(e));             \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        {                                               \
            ArrayMat4 m;                                \
            if (getArrayMat4(m))                        \
            {                                           \
                v.reserve(m.size());                    \
                for (auto &e : m)                       \
                {                                       \
                    v.push_back(Matrix(e));             \
                }                                       \
                return true;                            \
            }                                           \
        }                                               \
        return false;                                   \
    }                                                   \
    auto getArrayMatrix() const -> ArrayMatrix          \
    {                                                   \
        ArrayMatrix v;                                  \
        if (getArrayMatrix(v))                          \
        {                                               \
            return v;                                   \
        }                                               \
        else                                            \
        {                                               \
            return ArrayMatrix();                       \
        }                                               \
    }                                                   \
    Bool hasArrayMatrix() const                         \
    {                                                   \
        if (hasArrayMat2())                             \
        {                                               \
            return true;                                \
        }                                               \
        if (hasArrayMat3())                             \
        {                                               \
            return true;                                \
        }                                               \
        if (hasArrayMat4())                             \
        {                                               \
            return true;                                \
        }                                               \
        return false;                                   \
    }
#define HK_PROPERTY_OBJECT_DEFINE_ASSIGN(TYPE)                                                                                      \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                       \
    TYPE(const RefObjectT &v) : m_data(RefObjectUtils::convert<SRefObject>(v)) {}                                                \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                       \
    TYPE &operator=(const RefObjectT &v)                                                                                            \
    {                                                                                                                               \
        setObject(v);                                                                                                               \
        return *this;                                                                                                               \
    }                                                                                                                               \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr> \
    TYPE(const ArrayRefObjectT &v) : m_data() { setArrayObject(v); }                                                             \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr> \
    TYPE &operator=(ArrayRefObjectT &v)                                                                                             \
    {                                                                                                                               \
        setArrayObject(v);                                                                                                          \
        return *this;                                                                                                               \
    }

#define HK_PROPERTY_OBJECT_DEFINE()                                                                                                                                                              \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool getObject(RefObjectT &v) const                                                                                                                                                          \
    {                                                                                                                                                                                            \
        auto p = impl_getObject();                                                                                                                                                               \
        if (!p)                                                                                                                                                                                  \
        {                                                                                                                                                                                        \
            return false;                                                                                                                                                                        \
        }                                                                                                                                                                                        \
        v = RefObjectUtils::convert<RefObjectT>(p);                                                                                                                                              \
        return true;                                                                                                                                                                             \
    }                                                                                                                                                                                            \
    template <typename RefObjectT>                                                                                                                                                               \
    auto getObject() const -> std::remove_reference_t<decltype(enabler_t<std::is_base_of_v<Object, RefObjectT>>{nullptr}, std::declval<RefObjectT>())>                                           \
    {                                                                                                                                                                                            \
        RefObjectT v(nullptr);                                                                                                                                                                   \
        getObject(v);                                                                                                                                                                            \
        return v;                                                                                                                                                                                \
    }                                                                                                                                                                                            \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool setObject(const RefObjectT &v)                                                                                                                                                          \
    {                                                                                                                                                                                            \
        auto p = RefObjectUtils::convert<SRefObject>(v);                                                                                                                                         \
        if (!p)                                                                                                                                                                                  \
        {                                                                                                                                                                                        \
            return false;                                                                                                                                                                        \
        }                                                                                                                                                                                        \
        impl_setObject(p);                                                                                                                                                                       \
        return true;                                                                                                                                                                             \
    }                                                                                                                                                                                            \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool hasObject() const                                                                                                                                                                       \
    {                                                                                                                                                                                            \
        auto p = impl_getObject();                                                                                                                                                               \
        if (!p)                                                                                                                                                                                  \
        {                                                                                                                                                                                        \
            return false;                                                                                                                                                                        \
        }                                                                                                                                                                                        \
        return p.isConvertible(RefObjectT::type::kTypeString);                                                                                                                                   \
    }                                                                                                                                                                                            \
                                                                                                                                                                                                 \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool getValue(RefObjectT &v) const { return getObject(v); }                                                                                                                                  \
    template <typename RefObjectT>                                                                                                                                                               \
    auto getValue() const -> std::remove_reference_t<decltype(enabler_t<std::is_base_of_v<Object, RefObjectT>>{nullptr}, std::declval<RefObjectT>())>                                            \
    {                                                                                                                                                                                            \
        return getObject<RefObjectT>();                                                                                                                                                          \
    }                                                                                                                                                                                            \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool setValue(const RefObjectT &v)                                                                                                                                                           \
    {                                                                                                                                                                                            \
        return setObject(v);                                                                                                                                                                     \
    }                                                                                                                                                                                            \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                                                                                    \
    Bool hasValue() const                                                                                                                                                                        \
    {                                                                                                                                                                                            \
        return hasObject<RefObject>();                                                                                                                                                           \
    }                                                                                                                                                                                            \
                                                                                                                                                                                                 \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr>                                                              \
    Bool getArrayObject(ArrayRefObjectT &v) const                                                                                                                                                \
    {                                                                                                                                                                                            \
        auto p = impl_getArrayObject();                                                                                                                                                          \
        if (p.empty())                                                                                                                                                                           \
        {                                                                                                                                                                                        \
            return false;                                                                                                                                                                        \
        }                                                                                                                                                                                        \
        auto arr = ArrayRefObjectT();                                                                                                                                                            \
        arr.reserve(p.size());                                                                                                                                                                   \
        for (auto &elem : p)                                                                                                                                                                     \
        {                                                                                                                                                                                        \
            auto tmp = RefObjectUtils::convert<typename ArrayRefObjectT::value_type>(elem);                                                                                                      \
            auto ref = tmp.getRef();                                                                                                                                                             \
            if (!ref)                                                                                                                                                                            \
            {                                                                                                                                                                                    \
                return false;                                                                                                                                                                    \
            }                                                                                                                                                                                    \
            arr.push_back(tmp);                                                                                                                                                                  \
        }                                                                                                                                                                                        \
        v = arr;                                                                                                                                                                                 \
        return true;                                                                                                                                                                             \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT>                                                                                                                                                          \
    auto getArrayObject() const -> std::remove_reference_t<decltype(enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>>{nullptr}, std::declval<ArrayRefObjectT>())> \
    {                                                                                                                                                                                            \
        ArrayRefObjectT v;                                                                                                                                                                       \
        getArrayObject(v);                                                                                                                                                                       \
        return v;                                                                                                                                                                                \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename typename ArrayRefObjectT::value_type::type>> = nullptr>                                                     \
    Bool setArrayObject(const ArrayRefObjectT &v)                                                                                                                                                \
    {                                                                                                                                                                                            \
        auto arr = ArraySRefObject();                                                                                                                                                            \
        if (v.empty())                                                                                                                                                                           \
        {                                                                                                                                                                                        \
            impl_setArrayObject({});                                                                                                                                                             \
            return true;                                                                                                                                                                         \
        }                                                                                                                                                                                        \
        arr.reserve(v.size());                                                                                                                                                                   \
        for (auto &i : v)                                                                                                                                                                        \
        {                                                                                                                                                                                        \
            arr.push_back(RefObjectUtils::convert<SRefObject>(i));                                                                                                                               \
        }                                                                                                                                                                                        \
        impl_setArrayObject(arr);                                                                                                                                                                \
        return true;                                                                                                                                                                             \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr>                                                              \
    Bool hasArrayObject() const                                                                                                                                                                  \
    {                                                                                                                                                                                            \
        auto p = impl_getArrayObject();                                                                                                                                                          \
        if (p.empty())                                                                                                                                                                           \
        {                                                                                                                                                                                        \
            return false;                                                                                                                                                                        \
        }                                                                                                                                                                                        \
        for (auto &i : p)                                                                                                                                                                        \
        {                                                                                                                                                                                        \
            if (!i)                                                                                                                                                                              \
            {                                                                                                                                                                                    \
                continue;                                                                                                                                                                        \
            }                                                                                                                                                                                    \
            if (!i.isConvertible(ArrayRefObjectT::value_type::kTypeString))                                                                                                                      \
            {                                                                                                                                                                                    \
                return false;                                                                                                                                                                    \
            }                                                                                                                                                                                    \
        }                                                                                                                                                                                        \
        return true;                                                                                                                                                                             \
    }                                                                                                                                                                                            \
                                                                                                                                                                                                 \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr>                                                              \
    Bool getValue(ArrayRefObjectT &v) const                                                                                                                                                      \
    {                                                                                                                                                                                            \
        return getArrayObject(v);                                                                                                                                                                \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT>                                                                                                                                                          \
    auto getValue() const -> std::remove_reference_t<decltype(enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>>{nullptr}, std::declval<ArrayRefObjectT>())>       \
    {                                                                                                                                                                                            \
        return getArrayObject();                                                                                                                                                                 \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename typename ArrayRefObjectT::value_type::type>> = nullptr>                                                     \
    Bool setValue(const ArrayRefObjectT &v)                                                                                                                                                      \
    {                                                                                                                                                                                            \
        return setArrayObject(v);                                                                                                                                                                \
    }                                                                                                                                                                                            \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr>                                                              \
    Bool hasValue() const                                                                                                                                                                        \
    {                                                                                                                                                                                            \
        return hasArrayObject<ArrayRefObjectT>();                                                                                                                                                \
    }

#define HK_PROPERTY_ARRAY_TYPE_DEFINE(TYPE)                                     \
    HK_PROPERTY_ARRAY_TYPE_DEFINE_TYPE(TYPE);                                   \
    HK_PROPERTY_ARRAY_TYPE_DEFINE_VALU(TYPE);                                   \
    Bool has##Array##TYPE() const { return std::get_if<Array##TYPE>(&m_data); } \
    template <typename T, enabler_t<std::is_same_v<T, Array##TYPE>> = nullptr>  \
    Bool hasValue() const { return has##Array##TYPE(); }

#define HK_WREF_PROPERTY_TYPE_DEFINE(TYPE)                               \
    void operator=(const TYPE &v)                                        \
    {                                                                    \
        m_holder.setProperty(Property(v));                               \
        return;                                                          \
    }                                                                    \
    void set##TYPE(const TYPE &v) { m_holder.setProperty(Property(v)); } \
    Bool get##TYPE(TYPE &v) const                                        \
    {                                                                    \
        auto p = m_holder.getProperty();                                 \
        return p.get##TYPE(v);                                           \
    }                                                                    \
    auto get##TYPE() const -> Option<TYPE>                               \
    {                                                                    \
        auto p = m_holder.getProperty();                                 \
        return p.get##TYPE();                                            \
    }                                                                    \
    HK_PROPERTY_TYPE_DEFINE_VALU(TYPE)

#define HK_WREF_PROPERTY_ARRAY_TYPE_DEFINE(TYPE)                                       \
    void operator=(const Array##TYPE &v)                                               \
    {                                                                                  \
        m_holder.setProperty(Property(v));                                             \
        return;                                                                        \
    }                                                                                  \
    void set##Array##TYPE(const Array##TYPE &v) { m_holder.setProperty(Property(v)); } \
    Bool get##Array##TYPE(Array##TYPE &v) const                                        \
    {                                                                                  \
        auto p = m_holder.getProperty();                                               \
        return p.get##Array##TYPE(v);                                                  \
    }                                                                                  \
    auto get##Array##TYPE() const -> Array##TYPE                                       \
    {                                                                                  \
        auto p = m_holder.getProperty();                                               \
        return p.get##Array##TYPE();                                                   \
    }                                                                                  \
    HK_PROPERTY_ARRAY_TYPE_DEFINE_VALU(TYPE)

#define HK_WREF_PROPERTY_OBJECT_DEFINE_ASSIGN(TYPE)                                                                                 \
    template <typename RefObjectT, enabler_t<std::is_base_of_v<Object, typename RefObjectT::type>> = nullptr>                       \
    void operator=(const RefObjectT &v) { setObject(v); }                                                                           \
    template <typename ArrayRefObjectT, enabler_t<std::is_base_of_v<Object, typename ArrayRefObjectT::value_type::type>> = nullptr> \
    void operator=(ArrayRefObjectT &v) { setArrayObject(v); }
