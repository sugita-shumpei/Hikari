#include <hikari/core/data_type.h>
#include <nlohmann/json.hpp>

auto hikari::core::convertStringToVec2(const Str& v) -> Option<Vec2>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_array()) {
    try {
      auto arr = j.get<std::vector<F32>>();
      if (arr.size() == 2) { return Vec2(arr[0], arr[1]); }
      else { return std::nullopt; }
    }
    catch (...) {
      return std::nullopt;
    }
  }
  if (j.is_object()) {
    auto iter_value = j.find("value");
    if (iter_value != j.end()) {
      try {
        auto arr = j.get<std::vector<F32>>();
        if (arr.size() == 2) { return Vec2(arr[0], arr[1]); }
        else { return std::nullopt; }
      }
      catch (...) {
        return std::nullopt;
      }
    }
    Vec2 res = {};
    auto iter_x = j.find("x");
    if (iter_x != j.end()) { try { res.x = iter_x.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_y = j.find("y");
    if (iter_y != j.end()) { try { res.y = iter_y.value().get<F32>(); return std::nullopt; } catch (...) {} }
    return res;
  }
  return Option<Vec2>();
}

auto hikari::core::convertStringToVec3(const Str& v) -> Option<Vec3>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_array()) {
    try {
      auto arr = j.get<std::vector<F32>>();
      if (arr.size() == 3) { return Vec3(arr[0], arr[1], arr[2]); }
      else { return std::nullopt; }
    }
    catch (...) {
      return std::nullopt;
    }
  }
  if (j.is_object()) {
    auto iter_value = j.find("value");
    if (iter_value != j.end()) {
      try {
        auto arr = j.get<std::vector<F32>>();
        if (arr.size() == 3) { return Vec3(arr[0], arr[1], arr[2]); }
        else { return std::nullopt; }
      }
      catch (...) {
        return std::nullopt;
      }
    }
    Vec3 res = {};
    auto iter_x = j.find("x");
    if (iter_x != j.end()) { try { res.x = iter_x.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_y = j.find("y");
    if (iter_y != j.end()) { try { res.y = iter_y.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_z = j.find("z");
    if (iter_z != j.end()) { try { res.z = iter_z.value().get<F32>(); return std::nullopt; } catch (...) {} }
    return res;
  }
  return Option<Vec3>();
}

auto hikari::core::convertStringToVec4(const Str& v) -> Option<Vec4>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_array()) {
    try {
      auto arr = j.get<std::vector<F32>>();
      if (arr.size() == 4) { return Vec4(arr[0], arr[1], arr[2], arr[3]); }
      else { return std::nullopt; }
    }
    catch (...) {
      return std::nullopt;
    }
  }
  if (j.is_object()) {
    auto iter_value = j.find("value");
    if (iter_value != j.end()) {
      try {
        auto arr = j.get<std::vector<F32>>();
        if (arr.size() == 4) { return Vec4(arr[0], arr[1], arr[2], arr[3]); }
        else { return std::nullopt; }
      }
      catch (...) {
        return std::nullopt;
      }
    }
    Vec4 res = {};
    auto iter_x = j.find("x");
    if (iter_x != j.end()) { try { res.x = iter_x.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_y = j.find("y");
    if (iter_y != j.end()) { try { res.y = iter_y.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_z = j.find("z");
    if (iter_z != j.end()) { try { res.z = iter_z.value().get<F32>(); return std::nullopt; } catch (...) {} }
    auto iter_w = j.find("w");
    if (iter_w != j.end()) { try { res.w = iter_w.value().get<F32>(); return std::nullopt; } catch (...) {} }
    return res;
  }
  return Option<Vec4>();
}

auto hikari::core::convertStringToMat2(const Str& v) -> Option<Mat2>
{
    nlohmann::json j = nlohmann::json::parse(v);
    if (j.is_array()) {
      try {
        auto arr = j.get< std::vector<std::vector<F32>>>();
        if (arr.size() != 2) { return std::nullopt; }
        for (auto& elem : arr) { if (elem.size() != 2) { return std::nullopt; } }
        Mat2 res = {};
        for (size_t i = 0; i < 2; ++i) {
          for (size_t j = 0; j < 2; ++j) {
            res[i][j] = arr[i][j];
          }
        }
        return res;
      }
      catch (...) {
        return std::nullopt;
      }
    }
    return std::nullopt;
}

auto hikari::core::convertStringToMat3(const Str& v) -> Option<Mat3>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_array()) {
    try {
      auto arr = j.get< std::vector<std::vector<F32>>>();
      if (arr.size() != 3) { return std::nullopt; }
      for (auto& elem : arr) { if (elem.size() != 3) { return std::nullopt; } }
      Mat3 res = {};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          res[i][j] = arr[i][j];
        }
      }
      return res;
    }
    catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

auto hikari::core::convertStringToMat4(const Str& v) -> Option<Mat4>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_array()) {
    try {
      auto arr = j.get< std::vector<std::vector<F32>>>();
      if (arr.size() != 4) { return std::nullopt; }
      for (auto& elem : arr) { if (elem.size() != 4) { return std::nullopt; } }
      Mat4 res = {};
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          res[i][j] = arr[i][j];
        }
      }
      return res;
    }
    catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

auto hikari::core::convertStringToQuat(const Str& v) -> Option<Quat>
{
  nlohmann::json j = nlohmann::json::parse(v);
  if (j.is_object()) {
    auto iter_value = j.find("value");
    if (iter_value != j.end()) {
      if (iter_value.value().is_array()) {
        auto v4 = convertStringToVec4(iter_value.value().dump());
        if (v4 != std::nullopt) { return Quat(v4->w,v4->x, v4->y, v4->z); }
      }
      return std::nullopt;
    }
    auto iter_euler_angles= j.find("euler_angles");
    if (iter_euler_angles != j.end()) {
      if (iter_euler_angles.value().is_array()) {
        auto v3 = convertStringToVec3(iter_euler_angles.value().dump());
        if (v3 != std::nullopt) { return Quat(glm::radians(*v3)); }
      }
      return std::nullopt;
    }
  }
  auto v4 = convertStringToVec4(v);
  if (v4 != std::nullopt) { return Quat(v4->w,v4->x, v4->y, v4->z); }
  auto v3 = convertStringToVec3(v);
  if (v3 != std::nullopt) { return Quat(glm::radians(*v3)); }
  return Option<Quat>();
}


#define HK_CONVERT_JSON_STRING_TO_DEFINE(TYPE) \
auto hikari::core::convertJSONStringTo##TYPE(const Str& v) -> Option<TYPE> { \
  nlohmann::json j = nlohmann::json::parse(v); \
  if (!j.is_object()) { return std::nullopt; } \
  auto iter_type = j.find("type"); \
  if (iter_type==j.end()) { return std::nullopt; } \
  auto& type = iter_type.value(); \
  if (!type.is_string()){ return std::nullopt; } \
  if (type.get<std::string>() != hikari::Type2String<TYPE>::value) { return std::nullopt; } \
  auto res = convertStringTo##TYPE(v); \
  if (res) { return res;} \
  auto iter_value = j.find("value"); \
  if (iter_value==j.end()) { return std::nullopt; } \
  auto& value = iter_value.value(); \
  return  convertStringTo##TYPE(value.dump()); \
}

HK_CONVERT_JSON_STRING_TO_DEFINE(I8);
HK_CONVERT_JSON_STRING_TO_DEFINE(I16);
HK_CONVERT_JSON_STRING_TO_DEFINE(I32);
HK_CONVERT_JSON_STRING_TO_DEFINE(I64);
HK_CONVERT_JSON_STRING_TO_DEFINE(U8);
HK_CONVERT_JSON_STRING_TO_DEFINE(U16);
HK_CONVERT_JSON_STRING_TO_DEFINE(U32);
HK_CONVERT_JSON_STRING_TO_DEFINE(U64);
HK_CONVERT_JSON_STRING_TO_DEFINE(F32);
HK_CONVERT_JSON_STRING_TO_DEFINE(F64);
auto hikari::core::convertJSONStringToBool(const hikari::Str& v) -> hikari::Option<hikari::Bool> { return hikari::convertStringToBool(v); }
auto hikari::core::convertJSONStringToStr(const hikari::Str& v) -> hikari::Option<hikari::Str>   { return hikari::convertStringToStr(v); }
HK_CONVERT_JSON_STRING_TO_DEFINE(Vec2);
HK_CONVERT_JSON_STRING_TO_DEFINE(Vec3);
HK_CONVERT_JSON_STRING_TO_DEFINE(Vec4);
HK_CONVERT_JSON_STRING_TO_DEFINE(Mat2);
HK_CONVERT_JSON_STRING_TO_DEFINE(Mat3);
HK_CONVERT_JSON_STRING_TO_DEFINE(Mat4);
HK_CONVERT_JSON_STRING_TO_DEFINE(Quat);

#undef HK_CONVERT_JSON_STRING_TO_DEFINE
