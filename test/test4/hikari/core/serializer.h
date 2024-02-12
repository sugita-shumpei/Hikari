#pragma once
#include <nlohmann/json.hpp>
#include <hikari/core/object.h>
namespace hikari
{
    inline namespace core
    {
        // Serialize用のインターフェイス
        // Serialize対象のObjectは継承クラスを提供すること
        //
        struct ObjectSerializer
        {
            virtual ~ObjectSerializer() noexcept {}
            virtual auto getTypeString() const noexcept -> Str = 0;
            virtual auto eval(const std::shared_ptr<Object> &object) const -> Json = 0;
        };
        // Serialize用のSingleton
        // Serialize対象のObjectはここに登録しておくこと
        //
        struct ObjectSerializeManager
        {
            static auto getInstance() -> ObjectSerializeManager &
            {
                static ObjectSerializeManager manager;
                return manager;
            }
            auto serialize(const std::shared_ptr<Object> &object) const -> Json
            {
                if (!object)
                {
                    return Json();
                }
                auto iter = m_serializer.find(object->getTypeString());
                if (iter != m_serializer.end())
                {
                    return iter->second->eval(object);
                }
                return Json();
            }
            void add(const std::shared_ptr<ObjectSerializer> &serializer)
            {
                if (!serializer)
                {
                    return;
                }
                m_serializer[serializer->getTypeString()] = serializer;
            }
            ~ObjectSerializeManager() noexcept {}
            ObjectSerializeManager(const ObjectSerializeManager &) = delete;
            ObjectSerializeManager &operator=(const ObjectSerializeManager &) = delete;
            ObjectSerializeManager(ObjectSerializeManager &&) = delete;
            ObjectSerializeManager &operator=(const ObjectSerializeManager &&) = delete;

        private:
            ObjectSerializeManager() noexcept : m_serializer{} {}
            Dict<Str, std::shared_ptr<ObjectSerializer>> m_serializer;
        };
        // Property用のSerializer
        //
        //
        template <typename T>
        struct PropertyTypeSerializer
        {
            template <Bool CheckType = true>
            static auto eval(T val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<T>::value;
                    json["value"] = val;
                }
                else
                {
                    json = val;
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Bool>
        {
            template <Bool CheckType = true>
            static auto eval(Bool val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Bool>::value;
                    json["value"] = val;
                }
                else
                {
                    json = val;
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Str>
        {
            template <Bool CheckType = true>
            static auto eval(const Str &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Str>::value;
                    json["value"] = val;
                }
                else
                {
                    json = val;
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Vec2>
        {
            template <Bool CheckType = true>
            static auto eval(const Vec2 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Vec2>::value;
                    json["value"] = std::vector<F32>{val.x, val.y};
                }
                else
                {
                    json = std::vector<F32>{val.x, val.y};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Vec3>
        {
            template <Bool CheckType = true>
            static auto eval(const Vec3 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Vec3>::value;
                    json["value"] = std::vector<F32>{val.x, val.y, val.z};
                }
                else
                {
                    json = std::vector<F32>{val.x, val.y, val.z};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Vec4>
        {
            template <Bool CheckType = true>
            static auto eval(const Vec4 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Vec4>::value;
                    json["value"] = std::vector<F32>{val.x, val.y, val.z, val.w};
                }
                else
                {
                    json = std::vector<F32>{val.x, val.y, val.z, val.w};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Mat2>
        {
            template <Bool CheckType = true>
            static auto eval(const Mat2 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Mat2>::value;
                    json["value"] = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1]},
                        std::vector<F32>{val[1][0], val[1][1]}};
                }
                else
                {
                    json = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1]},
                        std::vector<F32>{val[1][0], val[1][1]}};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Mat3>
        {
            template <Bool CheckType = true>
            static auto eval(const Mat3 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Mat3>::value;
                    json["value"] = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1], val[0][2]},
                        std::vector<F32>{val[1][0], val[1][1], val[1][2]},
                        std::vector<F32>{val[2][0], val[2][1], val[2][2]}};
                }
                else
                {
                    json = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1], val[0][2]},
                        std::vector<F32>{val[1][0], val[1][1], val[1][2]},
                        std::vector<F32>{val[2][0], val[2][1], val[2][2]}};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Mat4>
        {
            template <Bool CheckType = true>
            static auto eval(const Mat4 &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Mat4>::value;
                    json["value"] = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1], val[0][2], val[0][3]},
                        std::vector<F32>{val[1][0], val[1][1], val[1][2], val[1][3]},
                        std::vector<F32>{val[2][0], val[2][1], val[2][2], val[2][3]},
                        std::vector<F32>{val[3][0], val[3][1], val[3][2], val[3][3]}};
                }
                else
                {
                    json = std::vector<std::vector<F32>>{
                        std::vector<F32>{val[0][0], val[0][1], val[0][2], val[0][3]},
                        std::vector<F32>{val[1][0], val[1][1], val[1][2], val[1][3]},
                        std::vector<F32>{val[2][0], val[2][1], val[2][2], val[2][3]},
                        std::vector<F32>{val[3][0], val[3][1], val[3][2], val[3][3]}};
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Quat>
        {
            template <Bool CheckType = true>
            static auto eval(const Quat &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Quat>::value;
                }
                json["value"] = std::vector<F32>{val.w, val.x, val.y, val.z};
                auto euler = glm::degrees(glm::eulerAngles(val));
                json["euler_angles"] = std::vector<F32>{euler.x, euler.y, euler.z};
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<std::shared_ptr<Object>>
        {
            template <Bool CheckType = true>
            static auto eval(const std::shared_ptr<Object> &val) -> Json
            {
                return ObjectSerializeManager::getInstance().serialize(val);
            }
        };
        template <>
        struct PropertyTypeSerializer<Transform>
        {
            template <Bool CheckType = true>
            static auto eval(const Transform &val) -> Json
            {
                Json json = {};
                if (CheckType)
                {
                    json["type"] = Type2String<Transform>::value;
                }
                if (val.getType() == TransformType::eTRS)
                {
                    json["position"] = PropertyTypeSerializer<Vec3>::eval<false>(*val.getPosition());
                    json["rotation"] = PropertyTypeSerializer<Quat>::eval<false>(*val.getRotation());
                    json["scale"] = PropertyTypeSerializer<Vec3>::eval<false>(*val.getScale());
                }
                else
                {
                    json["matrix"] = PropertyTypeSerializer<Mat4>::eval<false>(val.getMat());
                }
                return json;
            }
        };
        template <typename T>
        struct PropertyTypeSerializer<Array<T>>
        {
            template <Bool CheckType = true>
            static auto eval(const Array<T> &val) -> Json
            {
                Json json     = {};
                if constexpr (CheckType) {
                  json["type"]  = Type2String<Array<T>>::value;
                }
                auto arr = std::vector<Json>();
                for (const auto &elm : val)
                {
                    auto tmp = PropertyTypeSerializer<T>::eval<false>(elm);
                    arr.push_back(tmp);
                }
                if constexpr (CheckType) {
                  json["value"] = arr;
                }
                else {
                  json = arr;
                }

                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<Array<std::shared_ptr<Object>>>
        {
          template <Bool CheckType = true>
            static auto eval(const Array<std::shared_ptr<Object>> &val) -> Json
            {
                Json json = {};
                if constexpr (CheckType) {
                  json["type"] = "Array<Object>";
                }
                auto arr = std::vector<Json>();
                for (auto &elm : val)
                {
                    auto tmp = PropertyTypeSerializer<std::shared_ptr<Object>>::eval(elm);
                    arr.push_back(tmp);
                }
                if constexpr (CheckType) {
                  json["value"] = arr;
                }
                else {
                  json = arr;
                }
                return json;
            }
        };
        template <>
        struct PropertyTypeSerializer<std::monostate>
        {
            static auto eval(const std::monostate &val) -> Json
            {
                Json json = {};
                return json;
            }
        };
        struct PropertySerializer
        {
            static auto eval(const Property &prop) -> Json
            {
                auto json = std::visit([](const auto &p)
                                       { return PropertyTypeSerializer<std::remove_cv_t<std::remove_reference_t<decltype(p)>>>::eval(p); },
                                       prop.toVariant());
                return json;
            }
        };
        // Serializer
        //
        //
        template <typename T>
        struct Serializer;
        //
        //
        //
        template <typename T, std::enable_if_t<in_tuple<T, PropertyDataTypes>::value, nullptr_t> = nullptr>
        auto serialize(const T &v) -> Json
        {
            return PropertyTypeSerializer<T>::eval<false>(v);
        }
        template <typename T, std::enable_if_t<in_tuple<T, PropertyArrayDataTypes>::value, nullptr_t> = nullptr>
        auto serialize(const T &v) -> Json
        {
            return PropertyTypeSerializer<T>::eval<false>(v);
        }
        template <typename T, std::enable_if_t<ObjectWrapperTraits<T>::value, nullptr_t> = nullptr>
        auto serialize(const T &v) -> Json
        {
            return PropertyTypeSerializer<std::shared_ptr<Object>>::eval(v.getObject());
        }
        template <typename T, std::enable_if_t<ObjectWrapperArrayTraits<T>::value, nullptr_t> = nullptr>
        auto serialize(const T &v) -> Json
        {
            auto arr = Array<std::shared_ptr<Object>>();
            arr.reserve(v.size());
            std::transform(std::begin(v), std::end(v), std::back_inserter(arr), [](const auto &s)
                           { return std::static_pointer_cast<Object>(s.getObject()); });
            return PropertyTypeSerializer<Array<std::shared_ptr<Object>>>::eval<false>(arr);
        }
        inline auto serialize(const Property &v) -> Json
        {
            return PropertySerializer::eval(v);
        }
        inline auto serialize(const PropertyBlock &v) -> Json
        {
            Json json = {};
            auto keys = v.getKeys();
            for (auto &key : keys)
            {
                json[key] = serialize(v.getValue(key));
            }
            return json;
        }
        template <typename T>
        auto serialize(const Option<T> &v) -> Json
        {
            if (v)
            {
                return serialize(*v);
            }
            else
            {
                return Json(nullptr);
            }
        }
    }
}
