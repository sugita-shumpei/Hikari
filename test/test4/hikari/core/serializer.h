#pragma once
#include <nlohmann/json.hpp>
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // Serialize用のインターフェイス
    // Serialize対象のObjectは継承クラスを提供すること
    // 
    struct ObjectSerializer {
      virtual ~ObjectSerializer() noexcept {}
      virtual auto getTypeString() const noexcept -> Str = 0;
      virtual auto eval(const std::shared_ptr<Object>& object) const-> Json = 0;
    };
    // Serialize用のSingleton
    // Serialize対象のObjectはここに登録しておくこと
    // 
    struct ObjectSerializeManager {
      static auto getInstance() -> ObjectSerializeManager& {
        static ObjectSerializeManager manager;
        return manager;
      }
      auto serialize(const std::shared_ptr<Object>& object)const -> Json {
        if (!object) { return Json(); }
        auto iter = m_serializer.find(object->getTypeString());
        if (iter != m_serializer.end()) {
          return iter->second->eval(object);
        }
        return Json();
      }
      void add(const std::shared_ptr<ObjectSerializer>& serializer) {
        if (!serializer) { return; }
        m_serializer[serializer->getTypeString()] = serializer;
      }
      ~ObjectSerializeManager() noexcept {}
       ObjectSerializeManager(const ObjectSerializeManager&) = delete;
       ObjectSerializeManager& operator=(const ObjectSerializeManager&) = delete;
       ObjectSerializeManager(ObjectSerializeManager&&) = delete;
       ObjectSerializeManager& operator=(const ObjectSerializeManager&&) = delete;
    private:
      ObjectSerializeManager() noexcept : m_serializer{} {}
      Dict<Str, std::shared_ptr<ObjectSerializer>> m_serializer;
    };
    // 
    // 
    //
    template<typename T>
    struct PropertyTypeSerializer      {
      static auto eval(T val)-> Json {
        Json json = {};
        json["type"]  = Type2String<T>::value;
        json["value"] = std::to_string(val);
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Bool>   {
      static auto eval(Bool val)-> Json {
      Json json = {};
      json["type"]  = Type2String<Bool>::value;
      json["value"] = val?"true":"false";
      return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Str> {
      static auto eval(const Str& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Str>::value;
        json["value"] = "\"" + val + "\"";
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Vec2> {
      static auto eval(const Vec2& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Vec2>::value;
        json["value"] = std::vector<F32>{ val.x,val.y };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Vec3> {
      static auto eval(const Vec3& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Vec3>::value;
        json["value"] = std::vector<F32>{ val.x,val.y,val.z };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Vec4> {
      static auto eval(const Vec4& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Vec4>::value;
        json["value"] = std::vector<F32>{ val.x,val.y,val.z,val.w };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Mat2> {
      static auto eval(const Mat2& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Mat2>::value;
        json["value"] = std::vector<std::vector<F32>>{
          std::vector<F32>{val[0][0],val[0][1]},
          std::vector<F32>{val[1][0],val[1][1]}
        };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Mat3> {
      static auto eval(const Mat3& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Mat3>::value;
        json["value"] = std::vector<std::vector<F32>>{
          std::vector<F32>{val[0][0],val[0][1],val[0][2]},
          std::vector<F32>{val[1][0],val[1][1],val[1][2]},
          std::vector<F32>{val[2][0],val[2][1],val[2][2]}
        };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Mat4> {
      static auto eval(const Mat4& val)-> Json {
        Json json = {};
        json["type"] = Type2String<Mat4>::value;
        json["value"] = std::vector<std::vector<F32>>{
          std::vector<F32>{val[0][0],val[0][1],val[0][2],val[0][3]},
          std::vector<F32>{val[1][0],val[1][1],val[1][2],val[1][3]},
          std::vector<F32>{val[2][0],val[2][1],val[2][2],val[2][3]},
          std::vector<F32>{val[3][0],val[3][1],val[3][2],val[3][3]}
        };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Quat> {
      static auto eval(const Quat& val)-> Json {
        Json json = {};
        json["type"]  = Type2String<Quat>::value;
        json["value"] = std::vector<F32>{ val.w,val.x,val.y,val.z };
        static auto euler    = glm::degrees(glm::eulerAngles(val));
        json["euler_angles"] = std::vector<F32>{ euler.x,euler.y,euler.z };
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<std::shared_ptr<Object>> {
      static auto eval(const std::shared_ptr<Object>& val) -> Json {
        return ObjectSerializeManager::getInstance().serialize(val);
      }
    };
    template<>
    struct PropertyTypeSerializer<Transform> {
      static auto eval(const Transform& val)-> Json {
        Json json = {};
        json["type"]  = Type2String<Transform>::value;
        if (val.getType() == TransformType::eTRS) {
          json["position"] = PropertyTypeSerializer<Vec3>::eval(*val.getPosition());
          json["rotation"] = PropertyTypeSerializer<Quat>::eval(*val.getRotation());
          json["scale"]    = PropertyTypeSerializer<Vec3>::eval(*val.getScale());
          json["position"].erase("type");
          json["rotation"].erase("type");
          json["scale"].erase("type");
        }
        else {
          json["matrix"] = PropertyTypeSerializer<Mat4>::eval(val.getMat());
          json["matrix"].erase("type");
        }
        return json;
      }
    };
    template<typename T>
    struct PropertyTypeSerializer<Array<T>> {
      static auto eval(const Array<T>& val) -> Json {
        Json json     = {};
        json["type"]  = Type2String<Array<T>>::value;
        json["value"] = {};
        for (auto elm : val) {
          auto tmp = PropertyTypeSerializer<T>::eval(elm);
          tmp.erase("type");
          json["value"].push_back(tmp);
        }
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer<Array<std::shared_ptr<Object>>> {
      static auto eval(const Array<std::shared_ptr<Object>>& val) -> Json {
        Json json = {};
        json["type"]  = "Array<Object>";
        json["value"] = {};
        for (auto& elm : val) {
          auto tmp = PropertyTypeSerializer<std::shared_ptr<Object>>::eval(elm);
          json["value"].push_back(tmp);
        }
        return json;
      }
    };
    template<>
    struct PropertyTypeSerializer <std::monostate> {
      static auto eval(const std::monostate& val) -> Json {
        Json json = {};
        return json;
      }
    };
    struct PropertySerializer {
      static auto eval(const Property& prop) -> Json {
        auto json = std::visit([](const auto& p) {
          return PropertyTypeSerializer<std::remove_cv_t<std::remove_reference_t<decltype(p)>>>::eval(p);
        }, prop.toVariant());
        return json;
      }
    };
  }
}
