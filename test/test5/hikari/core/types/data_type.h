#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstdint>
#include <Imath/half.h>
#include <optional>
#include <string>
#include <vector>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif
#define HK_DATA_TYPE_DEFINE_INT(SIZE)  typedef int##SIZE##_t  I##SIZE
#define HK_DATA_TYPE_DEFINE_UINT(SIZE) typedef uint##SIZE##_t U##SIZE

    HK_DATA_TYPE_DEFINE_INT(8);
    HK_DATA_TYPE_DEFINE_INT(16);
    HK_DATA_TYPE_DEFINE_INT(32);
    HK_DATA_TYPE_DEFINE_INT(64);

    HK_DATA_TYPE_DEFINE_UINT(8);
    HK_DATA_TYPE_DEFINE_UINT(16);
    HK_DATA_TYPE_DEFINE_UINT(32);
    HK_DATA_TYPE_DEFINE_UINT(64);

    typedef Imath::half F16;
    typedef float       F32;
    typedef double      F64;

    typedef I32         Int;
    typedef F32         Float;
    typedef U64         UPtr;
    typedef I64         IPtr;
    typedef U8          Byte;
    typedef char        Char;
    typedef bool        Bool;
    typedef const char* CStr;
    typedef std::string String;

    template<typename T>
    using Array = std::vector<T>;
    // 本当はArray<Bool>を使いたいが, Array<Bool>はcontinious iterativeでないという欠陥あり
    typedef Array<Int>     ArrayInt;
    typedef Array<Float>   ArrayFloat;
    typedef Array<UPtr>    ArrayUPtr;
    typedef Array<IPtr>    ArrayIPtr;
    typedef Array<Byte>    ArrayByte;
    typedef Array<Char>    ArrayChar;
    typedef Array<Bool>    ArrayBool;
    typedef Array<String>  ArrayString;
    typedef Array<CStr>    ArrayCStr;

    template<typename T>
    using Option = std::optional<T>;

#if defined(__cplusplus)
  }
}
#endif
