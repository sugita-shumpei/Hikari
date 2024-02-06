#pragma once
#include <cstdint>
#include <string>
#include <optional>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <hikari/core/tuple.h>
namespace hikari {
  inline namespace core {
    using I8 = int8_t;
    using I16 = int16_t;
    using I32 = int32_t;
    using I64 = int64_t;

    using U8 = uint8_t;
    using U16 = uint16_t;
    using U32 = uint32_t;
    using U64 = uint64_t;

    using F32 = float;
    using F64 = double;

    using Bool = bool;
    using Char = char;
    using Byte = unsigned char;
    using Str  = std::string;

    using Vec2 = glm::vec2;
    using Vec3 = glm::vec3;
    using Vec4 = glm::vec4;

    using Mat2 = glm::mat2;
    using Mat3 = glm::mat3;
    using Mat4 = glm::mat4;

    using Quat = glm::quat;

    template<typename T>
    using Option = std::optional<T>;

    template<typename T>
    using Array  = std::vector<T>;

    using SignedIntegerTypes    = hikari::Tuple<I8, I16, I32, I64>;
    using UnsignedIntegerTypes  = hikari::Tuple<U8, U16, U32, U64>;
    using FloatingPointingTypes = hikari::Tuple<F32, F64>;
    using NumericTypes          = typename hikari::concat_tuple<SignedIntegerTypes, UnsignedIntegerTypes, FloatingPointingTypes>::type;
    using VectorTypes           = hikari::Tuple<Vec2, Vec3, Vec4>;
    using MatrixTypes           = hikari::Tuple<Mat2, Mat3, Mat4>;
    using NonNullTypes          = typename hikari::concat_tuple<NumericTypes, VectorTypes, MatrixTypes, std::tuple<Quat, Bool>>::type;
    using DataTypes             = typename hikari::concat_tuple<NonNullTypes, std::tuple<Str>>::type;

    template<typename T> struct Type2String;
    template<typename T>
    struct Type2String<Array<T>> :
      integer_sequence_to_str_data<
        typename concat_integer_sequence<
          std::integer_sequence<char,'A','r','r','a','y','<'>,
            typename str_data_to_integer_sequence<Type2String<T>>::type,
          std::integer_sequence<char,'>'>
        >::type
      >
    {};
    template<typename T>
    struct Type2String<Option<T>> :
      integer_sequence_to_str_data<
      typename concat_integer_sequence<
      std::integer_sequence<char, 'O', 'p', 't', 'i', 'o','n', '<'>,
      typename str_data_to_integer_sequence<Type2String<T>>::type,
      std::integer_sequence<char, '>'>
      >::type
      >
    {};

    template<> struct Type2String<I8>  { static inline constexpr char value[]  = "I8"  ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<I16> { static inline constexpr char value[]  = "I16" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<I32> { static inline constexpr char value[]  = "I32" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<I64> { static inline constexpr char value[]  = "I64" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<U8>  { static inline constexpr char value[]  = "U8"  ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<U16> { static inline constexpr char value[]  = "U16" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<U32> { static inline constexpr char value[]  = "U32" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<U64> { static inline constexpr char value[]  = "U64" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<F32> { static inline constexpr char value[]  = "F32" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<F64> { static inline constexpr char value[]  = "F64" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<Vec2> { static inline constexpr char value[] = "Vec2"; static inline constexpr size_t len = sizeof(value) - 1; };
    template<> struct Type2String<Vec3> { static inline constexpr char value[] = "Vec3"; static inline constexpr size_t len = sizeof(value) - 1; };
    template<> struct Type2String<Vec4> { static inline constexpr char value[] = "Vec4"; static inline constexpr size_t len = sizeof(value) - 1; };
    template<> struct Type2String<Mat2> { static inline constexpr char value[] = "Mat2"  ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<Mat3> { static inline constexpr char value[] = "Mat3" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<Mat4> { static inline constexpr char value[] = "Mat4" ; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<Quat> { static inline constexpr char value[] = "Quat"; static inline constexpr size_t len = sizeof(value) - 1; };
    template<> struct Type2String<Bool>{ static inline constexpr char value[]  = "Bool"; static inline constexpr size_t len = sizeof(value)-1; };
    template<> struct Type2String<Str> { static inline constexpr char value[]  = "Str" ; static inline constexpr size_t len = sizeof(value)-1; };

    template<typename T, std::enable_if_t<in_tuple<T, NumericTypes>::value, nullptr_t> = nullptr>
    inline auto convertToString(T    v) -> Str { return std::to_string(v); }
    inline auto convertToString(std::monostate v) -> Str { return "null"; }
    inline auto convertToString(Bool v) -> Str { return v ? "true" : "false"; }
    inline auto convertToString(Str  v) -> Str { return "\"" + v + "\""; }
    inline auto convertToString(Vec2 v) -> Str { return "[" + std::to_string(v.x)   + "," + std::to_string(v.y)   + "]"; }
    inline auto convertToString(Vec3 v) -> Str { return "[" + std::to_string(v.x)   + "," + std::to_string(v.y)   + "," + std::to_string(v.z)   + "]"; }
    inline auto convertToString(Vec4 v) -> Str { return "[" + std::to_string(v.x)   + "," + std::to_string(v.y)   + "," + std::to_string(v.z)   + +"," + std::to_string(v.w) + "]"; }
    inline auto convertToString(Mat2 v) -> Str { return "[" + convertToString(v[0]) + "," + convertToString(v[1]) + "]"; }
    inline auto convertToString(Mat3 v) -> Str { return "[" + convertToString(v[0]) + "," + convertToString(v[1]) + "," + convertToString(v[2]) + "]"; }
    inline auto convertToString(Mat4 v) -> Str { return "[" + convertToString(v[0]) + "," + convertToString(v[1]) + "," + convertToString(v[2]) + +"," + convertToString(v[3]) + "]"; }
    inline auto convertToString(Quat v) -> Str {
      auto quat_value = convertToString(Vec4(v.x, v.y, v.z, v.w));
      auto euler_value = convertToString(glm::degrees(glm::eulerAngles(v)));
      return "{ \"value\" : " + quat_value + " , \"euler_angles\" : " + euler_value + " }";
    }
    template<typename T>
    inline auto convertToString(Option<T> v) -> Str { return v ? convertToString(*v) : "null"; }
    template<typename T>
    inline auto convertToString(const std::vector<T>& v) -> Str {
      if (v.empty()) {
        return "[]";
      }
      else {
        Str res = "[ ";
        for (size_t i = 0; i < v.size();++i) {
          auto str = convertToString(v[i]);
          res += str;
          if (i != v.size()-1) {
            res += ", ";
          }
        }
        res += " ]";
        return res;
      }
    }

    inline auto convertStringToI64(const Str& v) -> Option<I64> { try {  return std::stoll(v); } catch(const std::exception& e){ return std::nullopt; } }
    inline auto convertStringToI32(const Str& v) -> Option<I32> { try { return std::stoi(v); } catch (const std::exception& e) { return std::nullopt; } }
    inline auto convertStringToI16(const Str& v) -> Option<I16> { auto i32 = convertStringToI32(v); if (i32) { if (*i32 <= std::numeric_limits<I16>::max()) { return (I16)*i32; } else { return std::nullopt; } } return std::nullopt; }
    inline auto convertStringToI8(const Str& v)  -> Option<I8 > { auto i32 = convertStringToI32(v); if (i32) { if (*i32 <= std::numeric_limits<I8 >::max()) { return (I8)*i32; } else { return std::nullopt; } }  return std::nullopt; }
    inline auto convertStringToU64(const Str& v) -> Option<U64> { try { return std::stoull(v); } catch (const std::exception& e) { return std::nullopt; } }
    inline auto convertStringToU32(const Str& v) -> Option<U32> { try { return std::stoul(v); } catch (const std::exception& e) { return std::nullopt; } }
    inline auto convertStringToU16(const Str& v) -> Option<U16> { auto u32 = convertStringToU32(v); if (u32) { if (*u32 <= std::numeric_limits<U16>::max()) { return (U16)*u32; } else { return std::nullopt; } }  return std::nullopt; }
    inline auto convertStringToU8 (const Str& v) -> Option<U8 > { auto u32 = convertStringToU32(v); if (u32) { if (*u32 <= std::numeric_limits<U8 >::max())  { return (U8)*u32; } else { return std::nullopt; } }  return std::nullopt; }
    inline auto convertStringToF32(const Str& v) -> Option<F32> { try { return std::stof(v); } catch (const std::exception& e) { return std::nullopt; } }
    inline auto convertStringToF64(const Str& v) -> Option<F64> { try { return std::stod(v); } catch (const std::exception& e) { return std::nullopt; } }
    inline auto convertStringToStr(const Str& v) -> Option<Str> {
      if (v.size() <= 1) { return std::nullopt; }
      if (v.front() != '\"') { return std::nullopt; }
      if (v.back()  != '\"') { return std::nullopt; }
      return v.substr(1, v.size() - 2);
    }
    inline auto convertStringToBool(const Str& v) -> Option<Bool> {
      if (v == "true" || v =="True") { return true; }
      if (v == "false" || v == "False") { return false; }
      return std::nullopt;
    }

    auto convertStringToVec2(const Str& v) -> Option<Vec2>;
    auto convertStringToVec3(const Str& v) -> Option<Vec3>;
    auto convertStringToVec4(const Str& v) -> Option<Vec4>;
    auto convertStringToMat2(const Str& v) -> Option<Mat2>;
    auto convertStringToMat3(const Str& v) -> Option<Mat3>;
    auto convertStringToMat4(const Str& v) -> Option<Mat4>;
    auto convertStringToQuat(const Str& v) -> Option<Quat>;

    template<typename T>
    struct ConvertFromStringTraits : std::false_type {
    };
#define HK_CONVERT_FROM_STRING_DEFINE(TYPE) \
    template<> struct ConvertFromStringTraits<TYPE> : std::true_type { static auto eval(const Str& str)->Option<TYPE>{ return convertStringTo##TYPE(str); } \
    }

    HK_CONVERT_FROM_STRING_DEFINE(I8 );
    HK_CONVERT_FROM_STRING_DEFINE(I16);
    HK_CONVERT_FROM_STRING_DEFINE(I32);
    HK_CONVERT_FROM_STRING_DEFINE(I64);
    HK_CONVERT_FROM_STRING_DEFINE(U8);
    HK_CONVERT_FROM_STRING_DEFINE(U16);
    HK_CONVERT_FROM_STRING_DEFINE(U32);
    HK_CONVERT_FROM_STRING_DEFINE(U64);
    HK_CONVERT_FROM_STRING_DEFINE(F32);
    HK_CONVERT_FROM_STRING_DEFINE(F64);
    HK_CONVERT_FROM_STRING_DEFINE(Bool);
    HK_CONVERT_FROM_STRING_DEFINE(Str);
    HK_CONVERT_FROM_STRING_DEFINE(Vec2);
    HK_CONVERT_FROM_STRING_DEFINE(Vec3);
    HK_CONVERT_FROM_STRING_DEFINE(Vec4);
    HK_CONVERT_FROM_STRING_DEFINE(Mat2);
    HK_CONVERT_FROM_STRING_DEFINE(Mat3);
    HK_CONVERT_FROM_STRING_DEFINE(Mat4);
    HK_CONVERT_FROM_STRING_DEFINE(Quat);

    template<typename T, std::enable_if_t<ConvertFromStringTraits<T>::value,std::nullptr_t> =nullptr>
    inline auto convertFromString(const Str& v) -> Option<T> { return ConvertFromStringTraits<T>::eval(v); }

    template<typename T, std::enable_if_t<in_tuple<T, concat_tuple<NumericTypes,VectorTypes, MatrixTypes>::type>::value, nullptr_t> = nullptr>
    inline auto convertToJSONString(T v)    -> Str { return  "{ \"type\" : \"" + Str(Type2String<T>::value) + "\" , \"value\" : " + convertToString(v) + " }"; }
    inline auto convertToJSONString(Bool v) -> Str { return convertToString(v); }
    inline auto convertToJSONString(Str  v) -> Str { return convertToString(v); }
    inline auto convertToJSONString(std::monostate v) -> Str { return "null"; }
    inline auto convertToJSONString(Quat v) -> Str {
      auto quat_value = convertToString(Vec4(v.x, v.y, v.z, v.w));
      auto euler_value= convertToString(glm::degrees(glm::eulerAngles(v)));
      return "{ \"type\" : \"Quat\", \"value\" : " + quat_value + " , \"euler_angles\" : " + euler_value + " }";
    }
    template<typename T>
    inline auto convertToJSONString(Option<T> v) -> Str { if (v) { return convertToJSONString(*v); } else { return "null"; } }

    template<typename T>
    struct ConvertFromJSONStringTraits : std::false_type {
    };

#define HK_CONVERT_JSON_STRING_TO_DEFINE(TYPE) \
    auto convertJSONStringTo##TYPE(const Str& v) -> Option<TYPE>; \
    template<> struct ConvertFromJSONStringTraits<TYPE> : std::true_type { static auto eval(const Str& str)->Option<TYPE>{ return convertJSONStringTo##TYPE(str); } \
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
    HK_CONVERT_JSON_STRING_TO_DEFINE(Bool);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Str);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Vec2);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Vec3);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Vec4);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Mat2);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Mat3);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Mat4);
    HK_CONVERT_JSON_STRING_TO_DEFINE(Quat);

    template<typename T, std::enable_if_t<ConvertFromJSONStringTraits<T>::value, std::nullptr_t> = nullptr>
    inline auto convertFromJSONString(const Str& v) -> Option<T> { return ConvertFromJSONStringTraits<T>::eval(v); }

  }
}
