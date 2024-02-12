#pragma once
#include <nlohmann/json.hpp>
#include <hikari/core/object.h>
namespace hikari
{
    inline namespace core
    {
        // Deserialize用のインターフェイス
        // Deserialize対象のObjectは継承クラスを提供すること
        //
        struct ObjectDeserializer
        {
            virtual ~ObjectDeserializer() noexcept {}
            virtual auto getTypeString() const noexcept -> Str = 0;
            virtual auto eval(const Json &json) const -> std::shared_ptr<Object> = 0;
        };
        // Deserialize用のSingleton
        // Deserialize対象のObjectはここに登録しておくこと
        //
        struct ObjectDeserializeManager
        {
            static auto getInstance() -> ObjectDeserializeManager &
            {
                static ObjectDeserializeManager manager;
                return manager;
            }
            auto deserialize(const Json &json) const -> std::shared_ptr<Object>
            {
                auto type = json.find("type");
                if (type == json.end())
                {
                    return nullptr;
                }
                if (!type.value().is_string())
                {
                    return nullptr;
                }
                auto iter = m_deserializer.find(type.value().get<std::string>());
                if (iter != m_deserializer.end())
                {
                    return iter->second->eval(json);
                }
                return nullptr;
            }
            void add(const std::shared_ptr<ObjectDeserializer> &serializer)
            {
                if (!serializer)
                {
                    return;
                }
                m_deserializer[serializer->getTypeString()] = serializer;
            }
            ~ObjectDeserializeManager() noexcept {}
            ObjectDeserializeManager(const ObjectDeserializeManager &) = delete;
            ObjectDeserializeManager &operator=(const ObjectDeserializeManager &) = delete;
            ObjectDeserializeManager(ObjectDeserializeManager &&) = delete;
            ObjectDeserializeManager &operator=(const ObjectDeserializeManager &&) = delete;

        private:
            ObjectDeserializeManager() noexcept : m_deserializer{} {}
            Dict<Str, std::shared_ptr<ObjectDeserializer>> m_deserializer;
        };
        // Property用のSerializer
        //
        //
        template <typename T>
        struct PropertyTypeDeserializer;
#define HK_PROPERTY_TYPE_DESERIALIZER_INTEGER_DEFINE(TYPE)                \
    template <>                                                           \
    struct PropertyTypeDeserializer<TYPE>                                 \
    {                                                                     \
        template <Bool CheckType = true>                                  \
        static auto eval(const Json &json) -> Option<TYPE>                \
        {                                                                 \
            if constexpr (CheckType)                                      \
            {                                                             \
                auto type = json.find("type");                            \
                if (type == json.end())                                   \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                if (!type.value().is_string())                            \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                if (type.value().get<Str>() != Type2String<TYPE>::value)  \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                auto value = json.find("value");                          \
                if (value == json.end())                                  \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                if (!value.value().is_number_integer())                   \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                return safe_numeric_cast<TYPE>(value.value().get<I64>()); \
            }                                                             \
            else                                                          \
            {                                                             \
                if (!json.is_number_integer())                            \
                {                                                         \
                    return std::nullopt;                                  \
                }                                                         \
                return safe_numeric_cast<TYPE>(json.get<I64>());          \
            }                                                             \
        }                                                                 \
    }
#define HK_PROPERTY_TYPE_DESERIALIZER_UINTEGER_DEFINE(TYPE)           \
    template <>                                                       \
    struct PropertyTypeDeserializer<TYPE>                             \
    {                                                                 \
        template <Bool CheckType = true>                              \
        static auto eval(const Json &json) -> Option<TYPE>            \
        {                                                             \
            if constexpr (!CheckType)                                 \
            {                                                         \
                if (json.is_number_integer())                         \
                {                                                     \
                    return safe_numeric_cast<TYPE>(json.get<U64>());  \
                }                                                     \
            }                                                         \
            auto type = json.find("type");                            \
            if (type == json.end())                                   \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (!type.value().is_string())                            \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (type.value().get<Str>() != Type2String<TYPE>::value)  \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            auto value = json.find("value");                          \
            if (value == json.end())                                  \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (!value.value().is_number_integer())                   \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            return safe_numeric_cast<TYPE>(value.value().get<U64>()); \
        }                                                             \
    }
        template <typename T>
        struct PropertyTypeDeserializer;
#define HK_PROPERTY_TYPE_DESERIALIZER_FLOAT_DEFINE(TYPE)              \
    template <>                                                       \
    struct PropertyTypeDeserializer<TYPE>                             \
    {                                                                 \
        template <Bool CheckType = true>                              \
        static auto eval(const Json &json) -> Option<TYPE>            \
        {                                                             \
            if constexpr (!CheckType)                                 \
            {                                                         \
                if (!json.is_number())                                \
                {                                                     \
                    return std::nullopt;                              \
                }                                                     \
                return safe_numeric_cast<TYPE>(json.get<F64>());      \
            }                                                         \
            auto type = json.find("type");                            \
            if (type == json.end())                                   \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (!type.value().is_string())                            \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (type.value().get<Str>() != Type2String<TYPE>::value)  \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            auto value = json.find("value");                          \
            if (value == json.end())                                  \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            if (!value.value().is_number())                           \
            {                                                         \
                return std::nullopt;                                  \
            }                                                         \
            return safe_numeric_cast<TYPE>(value.value().get<F64>()); \
        }                                                             \
    }

        HK_PROPERTY_TYPE_DESERIALIZER_INTEGER_DEFINE(I8);
        HK_PROPERTY_TYPE_DESERIALIZER_INTEGER_DEFINE(I16);
        HK_PROPERTY_TYPE_DESERIALIZER_INTEGER_DEFINE(I32);
        HK_PROPERTY_TYPE_DESERIALIZER_INTEGER_DEFINE(I64);
        HK_PROPERTY_TYPE_DESERIALIZER_UINTEGER_DEFINE(U8);
        HK_PROPERTY_TYPE_DESERIALIZER_UINTEGER_DEFINE(U16);
        HK_PROPERTY_TYPE_DESERIALIZER_UINTEGER_DEFINE(U32);
        HK_PROPERTY_TYPE_DESERIALIZER_UINTEGER_DEFINE(U64);
        HK_PROPERTY_TYPE_DESERIALIZER_FLOAT_DEFINE(F16);
        HK_PROPERTY_TYPE_DESERIALIZER_FLOAT_DEFINE(F32);
        HK_PROPERTY_TYPE_DESERIALIZER_FLOAT_DEFINE(F64);
        template <>
        struct PropertyTypeDeserializer<Bool>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Bool>
            {
                auto type = json.find("type");
                if constexpr (!CheckType)
                {
                    if (json.is_boolean())
                    {
                        return json.get<bool>();
                    }
                }
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Bool>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (!value.value().is_boolean())
                {
                    return std::nullopt;
                }
                return value.value().get<Bool>();
            }
        };
        template <>
        struct PropertyTypeDeserializer<Str>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Str>
            {
                if constexpr (!CheckType)
                {
                    if (json.is_string())
                    {
                        return json.get<Str>();
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Str>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (!value.value().is_string())
                {
                    return std::nullopt;
                }
                return value.value().get<Str>();
            }
        };
        template <>
        struct PropertyTypeDeserializer<Vec2>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Vec2>
            {
                if constexpr (!CheckType)
                {
                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Vec2(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<F32, 2>>();
                            return Vec2(arr[0], arr[1]);
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Vec2>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Vec2(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<F32, 2>>();
                        return Vec2(arr[0], arr[1]);
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
            }
        };
        template <>
        struct PropertyTypeDeserializer<Vec3>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Vec3>
            {
                auto type = json.find("type");
                if constexpr (CheckType)
                {
                    if (type == json.end())
                    {
                        return std::nullopt;
                    }
                    if (!type.value().is_string())
                    {
                        return std::nullopt;
                    }
                    if (type.value().get<Str>() != Type2String<Vec3>::value)
                    {
                        return std::nullopt;
                    }
                }
                else
                {
                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Vec3(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<F32, 3>>();
                            return Vec3(arr[0], arr[1], arr[2]);
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Vec3(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<F32, 3>>();
                        return Vec3(arr[0], arr[1], arr[2]);
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Vec4>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Vec4>
            {
                if constexpr (!CheckType)
                {
                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Vec4(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<F32, 4>>();
                            if (arr.size() != 4)
                            {
                                return std::nullopt;
                            }
                            return Vec4(arr[0], arr[1], arr[2], arr[3]);
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Vec4>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Vec4(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<F32, 4>>();
                        if (arr.size() != 4)
                        {
                            return std::nullopt;
                        }
                        return Vec4(arr[0], arr[1], arr[2], arr[3]);
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Mat2>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Mat2>
            {
                if constexpr (!CheckType)
                {
                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Mat2(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<std::array<F32, 2>, 2>>();
                            return Mat2(Vec2(arr[0][0], arr[0][1]), Vec2(arr[1][0], arr[1][1]));
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Mat2>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Mat2(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<std::array<F32, 2>, 2>>();
                        return Mat2(Vec2(arr[0][0], arr[0][1]), Vec2(arr[1][0], arr[1][1]));
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Mat3>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Mat3>
            {
                if constexpr (!CheckType)
                {

                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Mat3(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<std::array<F32, 3>, 3>>();
                            return Mat3(Vec3(arr[0][0], arr[0][1], arr[0][2]), Vec3(arr[1][0], arr[1][1], arr[1][2]), Vec3(arr[2][0], arr[2][1], arr[2][2]));
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Mat3>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Mat3(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<std::array<F32, 3>, 3>>();
                        return Mat3(Vec3(arr[0][0], arr[0][1], arr[0][2]), Vec3(arr[1][0], arr[1][1], arr[1][2]), Vec3(arr[2][0], arr[2][1], arr[2][2]));
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Mat4>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Mat4>
            {
                if constexpr (!CheckType)
                {

                    if (json.is_number())
                    {
                        auto v = safe_numeric_cast<F32>(json.get<F64>());
                        if (v)
                        {
                            return Mat4(*v);
                        }
                        return std::nullopt;
                    }
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::array<std::array<F32, 4>, 4>>();
                            return Mat4(Vec4(arr[0][0], arr[0][1], arr[0][2], arr[0][3]), Vec4(arr[1][0], arr[1][1], arr[1][2], arr[1][3]), Vec4(arr[2][0], arr[2][1], arr[2][2], arr[2][3]), Vec4(arr[3][0], arr[3][1], arr[3][2], arr[3][3]));
                        }
                        catch (const std::exception &)
                        {
                            return std::nullopt;
                        }
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return std::nullopt;
                }
                if (!type.value().is_string())
                {
                    return std::nullopt;
                }
                if (type.value().get<Str>() != Type2String<Mat4>::value)
                {
                    return std::nullopt;
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return std::nullopt;
                }
                if (value.value().is_number())
                {
                    auto v = safe_numeric_cast<F32>(value.value().get<F64>());
                    if (v)
                    {
                        return Mat4(*v);
                    }
                    return std::nullopt;
                }
                if (value.value().is_array())
                {
                    try
                    {
                        auto arr = value.value().get<std::array<std::array<F32, 4>, 4>>();
                        return Mat4(Vec4(arr[0][0], arr[0][1], arr[0][2], arr[0][3]), Vec4(arr[1][0], arr[1][1], arr[1][2], arr[1][3]), Vec4(arr[2][0], arr[2][1], arr[2][2], arr[2][3]), Vec4(arr[3][0], arr[3][1], arr[3][2], arr[3][3]));
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Quat>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Quat>
            {
                if constexpr (!CheckType)
                {
                    if (json.is_array())
                    {
                        try
                        {
                            auto arr = json.get<std::vector<F32>>();
                            if (arr.size() == 3)
                            {
                                return Quat(glm::radians(Vec3(arr[0], arr[1], arr[2])));
                            }
                            if (arr.size() == 4)
                            {
                                return Quat(arr[0], arr[1], arr[2], arr[3]);
                            }
                        }
                        catch (...)
                        {
                        }
                    }
                }
                auto type = json.find("type");

                if (type != json.end())
                {
                  if (!type.value().is_string())
                  {
                    return std::nullopt;
                  }
                  if (type.value().get<Str>() != Type2String<Quat>::value)
                  {
                    return std::nullopt;
                  }
                }
                
                auto quat_value = json.find("value");
                auto euler_value = json.find("euler_angles");
                if (quat_value != json.end())
                {
                    if (!quat_value.value().is_array())
                    {
                        return std::nullopt;
                    }
                    try
                    {
                        auto arr = quat_value.value().get<std::array<F32, 4>>();
                        return Quat(arr[0], arr[1], arr[2], arr[3]);
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                if (euler_value != json.end())
                {
                    if (!euler_value.value().is_array())
                    {
                        return std::nullopt;
                    }
                    try
                    {
                        auto arr = euler_value.value().get<std::array<F32, 3>>();
                        return Quat(glm::radians(Vec3(arr[0], arr[1], arr[2])));
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<Transform>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Option<Transform>
            {
                auto type = json.find("type");
                if constexpr (CheckType)
                {
                    if (type == json.end())
                    {
                        return std::nullopt;
                    }
                    if (!type.value().is_string())
                    {
                        return std::nullopt;
                    }
                    if (type.value().get<Str>() != Type2String<Transform>::value)
                    {
                        return std::nullopt;
                    }
                }
                Transform tran;
                auto matrix_value = json.find("matrix");
                if (matrix_value != json.end())
                {
                    auto mat1 = PropertyTypeDeserializer<F32>::eval<false>(matrix_value.value());
                    if (mat1)
                    {
                        tran.setMat(Mat4(*mat1));
                        return tran;
                    }
                    auto mat2 = PropertyTypeDeserializer<Mat2>::eval<false>(matrix_value.value());
                    if (mat2)
                    {
                        tran.setMat(Mat4(*mat2));
                        return tran;
                    }
                    auto mat3 = PropertyTypeDeserializer<Mat3>::eval<false>(matrix_value.value());
                    if (mat3)
                    {
                        tran.setMat(Mat4(*mat3));
                        return tran;
                    }
                    auto mat4 = PropertyTypeDeserializer<Mat4>::eval<false>(matrix_value.value());
                    if (mat4)
                    {
                        tran.setMat(Mat4(*mat4));
                        return tran;
                    }
                    return std::nullopt;
                }
                TransformTRSData trs;
                bool is_ok = false;
                auto position_value = json.find("position");
                if (position_value != json.end())
                {
                    auto vec3 = PropertyTypeDeserializer<Vec3>::eval<false>(position_value.value());
                    if (!vec3)
                    {
                        return std::nullopt;
                    }
                    trs.position = *vec3;
                    is_ok = true;
                }
                auto rotation_value = json.find("rotation");
                if (rotation_value != json.end())
                {
                    auto quat = PropertyTypeDeserializer<Quat>::eval<false>(rotation_value.value());
                    if (!quat)
                    {
                        return std::nullopt;
                    }
                    trs.rotation = *quat;
                    is_ok = true;
                }
                auto scale_value = json.find("scale");
                if (scale_value != json.end())
                {
                    auto vec3 = PropertyTypeDeserializer<Vec3>::eval<false>(scale_value.value());
                    if (!vec3)
                    {
                        return std::nullopt;
                    }
                    trs.scale = *vec3;
                    is_ok = true;
                }
                return is_ok ? Option<Transform>(Transform(trs)) : std::nullopt;
            }
        };
        template <>
        struct PropertyTypeDeserializer<std::shared_ptr<Object>>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> std::shared_ptr<Object>
            {
                return ObjectDeserializeManager::getInstance().deserialize(json);
            }
        };
        template <>
        struct PropertyTypeDeserializer<std::vector<std::shared_ptr<Object>>>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> std::vector<std::shared_ptr<Object>>
            {

                if constexpr (!CheckType)
                {
                    if (json.is_array())
                    {
                        auto arr = json.get<Array<Json>>();
                        Array<std::shared_ptr<Object>> t = {};
                        for (auto &elm : arr)
                        {
                            auto tmp = PropertyTypeDeserializer<std::shared_ptr<Object>>::eval(elm);
                            if (!tmp)
                            {
                                return {};
                            }
                            t.push_back(tmp);
                        }
                        return t;
                    }
                }
                auto type = json.find("type");
                if (type == json.end())
                {
                    return {};
                }
                if (!type.value().is_string())
                {
                    return {};
                }
                if (type.value().get<Str>() != "Array<Object>")
                {
                    return {};
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return {};
                }
                if (!value.value().is_array())
                {
                    return {};
                }
                auto arr = value.value().get<Array<Json>>();
                Array<std::shared_ptr<Object>> t = {};
                for (auto &elm : arr)
                {
                    auto tmp = PropertyTypeDeserializer<std::shared_ptr<Object>>::eval(elm);
                    if (!tmp)
                    {
                        return {};
                    }
                    t.push_back(tmp);
                }
                return t;
            }
        };
        template <typename T>
        struct PropertyTypeDeserializer<Array<T>>
        {
            template <Bool CheckType = true>
            static auto eval(const Json &json) -> Array<T>
            {
                auto type = json.find("type");
                if constexpr (!CheckType)
                {
                    if (json.is_array())
                    {
                        auto arr = json.get<std::vector<Json>>();
                        Array<T> t = {};
                        for (const auto &elm : arr)
                        {
                            auto tmp = PropertyTypeDeserializer<T>::eval<false>(elm);
                            if (!tmp)
                            {
                                return {};
                            }
                            t.push_back(*tmp);
                        }
                        return t;
                    }
                }
                if (type == json.end())
                {
                    return {};
                }
                if (!type.value().is_string())
                {
                    return {};
                }
                if (type.value().get<Str>() != Type2String<Array<T>>::value)
                {
                    return {};
                }
                auto value = json.find("value");
                if (value == json.end())
                {
                    return {};
                }
                if (!value.value().is_array())
                {
                    return {};
                }
                auto arr = value.value().get<Array<Json>>();
                Array<T> t = {};
                for (const auto &elm : arr)
                {
                    auto tmp = PropertyTypeDeserializer<T>::eval<false>(elm);
                    if (!tmp)
                    {
                        return {};
                    }
                    t.push_back(*tmp);
                }
                return t;
            }
        };

        template <typename T, std::enable_if_t<in_tuple<T, PropertyDataTypes>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> Option<T>
        {
            return PropertyTypeDeserializer<T>::eval<false>(v);
        }
        template <typename T, std::enable_if_t<in_tuple<T, PropertyArrayDataTypes>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> T
        {
            return PropertyTypeDeserializer<T>::eval<false>(v);
        }
        template <typename T, std::enable_if_t<ObjectWrapperTraits<T>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> T
        {
            auto object = PropertyTypeDeserializer<std::shared_ptr<Object>>::eval(v);
            return ObjectWrapperUtils::convert<T>(ObjectWrapper(object));
        }
        template <typename T, std::enable_if_t<ObjectWrapperArrayTraits<T>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> T
        {
            using value_type = std::remove_reference_t<std::remove_cv_t<decltype(std::declval<T &>()[0])>>;
            auto objects = PropertyTypeDeserializer<Array<std::shared_ptr<Object>>>::eval<false>(v);
            auto arr = T();
            for (const auto &elm : objects)
            {
                if (!elm)
                {
                    arr.push_back(value_type(nullptr));
                    continue;
                }
                if (!elm->isConvertible(value_type::type::TypeString()))
                {
                    return {};
                }
                arr.push_back(value_type(std::static_pointer_cast<typename value_type::type>(elm)));
            }
            return arr;
        }
        template <typename T, std::enable_if_t<std::is_same<T, Property>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> T
        {
            T prop = {};
            {
                auto tmp = PropertyTypeDeserializer<I8>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<I16>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<I32>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<I64>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<U8>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<U16>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<U32>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<U64>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<F32>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<F64>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Vec2>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Vec3>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Vec4>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Mat2>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Mat3>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Mat4>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Quat>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Str>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Transform>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Bool>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<std::shared_ptr<Object>>::eval(v);
                if (tmp)
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<I8>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<I16>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<I32>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<I64>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<U8>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<U16>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<U32>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<U64>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<F32>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<F64>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Vec2>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Vec3>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Vec4>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Mat2>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Mat3>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Mat4>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Quat>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Str>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Transform>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<Bool>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            {
                auto tmp = PropertyTypeDeserializer<Array<std::shared_ptr<Object>>>::eval(v);
                if (!tmp.empty())
                {
                    prop = tmp;
                    return prop;
                }
            }
            return prop;
        }
        template <typename T, std::enable_if_t<std::is_same<T, PropertyBlock>::value, nullptr_t> = nullptr>
        auto deserialize(const Json &v) -> T
        {
            if (!v.is_object())
            {
                return PropertyBlock();
            }
            auto items = v.items();
            PropertyBlock pb;
            for (auto &item : items)
            {
                auto prop = deserialize<Property>(item.value());
                pb.setValue(item.key(), prop);
            }
            return pb;
        }
    }
}
