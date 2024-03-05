#pragma once
#include <Imath/half.h>
#include <string>
#include <cstdint>
#define HK_SIGNED_TYPE_DEFINE(SIZE) \
  using I##SIZE = std::int##SIZE##_t

#define HK_UNSIGNED_TYPE_DEFINE(SIZE) \
  using U##SIZE = std::uint##SIZE##_t

#define HK_CHAR_TYPE_DEFINE(SIZE) \
  using U##SIZE = std::uint##SIZE##_t

namespace hikari {
  namespace core {
    HK_SIGNED_TYPE_DEFINE(8);
    HK_SIGNED_TYPE_DEFINE(16);
    HK_SIGNED_TYPE_DEFINE(32);
    HK_SIGNED_TYPE_DEFINE(64);
    HK_UNSIGNED_TYPE_DEFINE(8);
    HK_UNSIGNED_TYPE_DEFINE(16);
    HK_UNSIGNED_TYPE_DEFINE(32);
    HK_UNSIGNED_TYPE_DEFINE(64);
    HK_CHAR_TYPE_DEFINE(8);
    HK_CHAR_TYPE_DEFINE(16);
    HK_CHAR_TYPE_DEFINE(32);
    using F16   = Imath::half;
    using F32   = float;
    using F64   = double;
    using Int   = I32;
    using UInt  = U32;
    using Float = F32;
    using Char  = char;
    using WChar = wchar_t;
    using Byte  = unsigned char;
    using CStr  = const Char*;
    using String= std::string;
    using WString = std::wstring;

    template<typename T>
    struct Extent2 { T width; T height; };
    template<typename T>
    struct Offset2 { T x; T y; };
  }
}
