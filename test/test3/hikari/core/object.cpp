#include <hikari/core/object.h>
#include <hikari/core/data_type.h>

auto hikari::core::Property::getString() const -> Str
{
  return std::visit([](const auto& v) { return convertPropertyTypeToString(v); }, m_data);
}

auto hikari::core::Property::getJSONString() const -> Str {
   return std::visit([](const auto& v) { return convertPropertyTypeToJSONString(v); }, m_data);
 }

 auto hikari::core::Property::getInteger() const -> Option<I64>
 {
   { auto value = getValue<I8>() ; if (value) { return *value; }}
   { auto value = getValue<I16>(); if (value) { return *value; }}
   { auto value = getValue<I32>(); if (value) { return *value; }}
   { auto value = getValue<I64>(); if (value) { return *value; }}
   { auto value = getValue<U8>() ; if (value) { return *value; }}
   { auto value = getValue<U16>(); if (value) { return *value; }}
   { auto value = getValue<U32>(); if (value) { return *value; }}
   { auto value = getValue<U64>(); if (value) { return *value >= std::numeric_limits<I64>::max() ? std::nullopt : Option<I64>((I64)*value); }}
   return Option<I64>();
 }

 auto hikari::core::Property::getFloat() const -> Option<F64>
 {
   { auto value = getValue<F32>(); if (value) { return *value; }}
   { auto value = getValue<F64>(); if (value) { return *value; }}
   { auto value = getValue<I8>(); if (value) { return *value; }}
   { auto value = getValue<I16>(); if (value) { return *value; }}
   { auto value = getValue<I32>(); if (value) { return *value; }}
   { auto value = getValue<I64>(); if (value) { return *value; }}
   { auto value = getValue<U8>(); if (value) { return *value; }}
   { auto value = getValue<U16>(); if (value) { return *value; }}
   { auto value = getValue<U32>(); if (value) { return *value; }}
   { auto value = getValue<U64>(); if (value) { return *value; }}
   return Option<F64>();
 }

 auto hikari::core::Property::getVector() const -> Option<Vec4>
 {
   { auto value = getValue<Vec2>(); if (value) { return Vec4(*value,0.0f,0.0f); }}
   { auto value = getValue<Vec3>(); if (value) { return Vec4(*value,0.0f); }}
   { auto value = getValue<Vec4>(); if (value) { return Vec4(*value); }}
   return Option<Vec4>();
 }

 auto hikari::core::Property::getMatrix() const -> Option<Mat4>
 {
   { auto value = getValue<Mat2>(); if (value) { return Mat2(*value); }}
   { auto value = getValue<Mat3>(); if (value) { return Mat3(*value); }}
   { auto value = getValue<Mat4>(); if (value) { return Mat4(*value); }}
   return Option<Mat4>();
 }

 auto hikari::core::Property::getIntegers() const -> std::vector<hikari::I64>
 {
   { auto value = getValue<std::vector<I8>>() ; if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<I16>>(); if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<I32>>(); if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<I64>>(); if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<U8>>() ; if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<U16>>(); if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
   { auto value = getValue<std::vector<U32>>(); if (!value.empty()) { std::vector<I64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res;}}
    auto value = getValue<std::vector<U64>>(); if (!value.empty()) {
     for (auto& elm : value) {
       if (elm >= std::numeric_limits<I64>::max()) {
         return {};
       }
     }
     std::vector<I64> res(value.size());
     std::copy(value.begin(), value.end(), res.begin());
     return res;
   }
   return std::vector<I64>();

 }
 auto hikari::core::Property::getFloats() const -> std::vector<F64>
 {
   { auto value = Property::getValue<std::vector<I8>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<I16>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<I32>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<I64>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<U8>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<U16>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<U32>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<U64>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<F32>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   { auto value = Property::getValue<std::vector<F64>>(); if (!value.empty()) { std::vector<F64> res(value.size()); std::copy(value.begin(), value.end(), res.begin()); return res; }}
   return std::vector<F64>();
 }
 auto hikari::core::Property::getVectors() const -> std::vector<Vec4>
 {
   { auto value = Property::getValue<std::vector<Vec2>>(); if (!value.empty()) { std::vector<Vec4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Vec4(v, 0.0f, 0.0f); }); return res; }}
   { auto value = Property::getValue<std::vector<Vec3>>(); if (!value.empty()) { std::vector<Vec4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Vec4(v, 0.0f); }); return res; }}
   { auto value = Property::getValue<std::vector<Vec4>>(); if (!value.empty()) { std::vector<Vec4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Vec4(v); }); return res; }}
   return std::vector<Vec4>();
 }
 auto hikari::core::Property::getMatrices() const -> std::vector<Mat4>
 {
   { auto value = Property::getValue<std::vector<Mat2>>(); if (!value.empty()) { std::vector<Mat4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Mat4(v); }); return res; }}
   { auto value = Property::getValue<std::vector<Mat3>>(); if (!value.empty()) { std::vector<Mat4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Mat4(v); }); return res; }}
   { auto value = Property::getValue<std::vector<Mat4>>(); if (!value.empty()) { std::vector<Mat4> res(value.size()); std::transform(value.begin(), value.end(), res.begin(), [](const auto& v) { return Mat4(v); }); return res; }}
   return std::vector<Mat4>();
 }
 auto hikari::core::convertPropertyToString(const Property& prop) -> Str
 {
   return prop.getString();
 }
 auto hikari::core::convertPropertyToJSONString(const Property& prop) -> Str
 {
   return prop.getJSONString();
 }

 auto hikari::core::convertJSONStringToProperty(const Str& str) -> Property
 {
#define HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(TYPE) \
   { \
   auto ret = convertJSONStringToPropertyType<TYPE>(str); \
    if (ret) { return Property(*ret); } \
   }
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(I8);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(I16);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(I32);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(I64);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(U8);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(U16);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(U32);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(U64);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(F32);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(F64);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Vec2);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Vec3);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Vec4);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Mat2);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Mat3);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Mat4);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Quat);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Transform);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Bool);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(Str);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<I8>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<I16>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<I32>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<I64>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<U8>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<U16>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<U32>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<U64>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<F32>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<F64>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Vec2>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Vec3>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Vec4>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Mat2>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Mat3>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Mat4>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Quat>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Transform>);

   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Bool>);
   HK_CONVERT_JSON_STRING_TO_PROPERTY_CASE(std::vector<Str>);

   return Property();
 }
