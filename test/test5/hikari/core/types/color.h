#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <hikari/core/types/data_type.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    struct Color {
      Color() noexcept {}
      F32 r = 0.0f;
      F32 g = 0.0f;
      F32 b = 0.0f;
      F32 a = 0.0f;
    };
    struct Color32 {
      U8  r;
      U8  g;
      U8  b;
      U8  a;
    };

    typedef Array<Color> ArrayColor;

#if defined(__cplusplus)
  }
}
#endif
