#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <hikari/core/types/vector.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    template<typename VectorT>
    struct AABBT {
      constexpr auto getCenter() const noexcept -> VectorT { return (min + max) / 2; }
      constexpr auto getRange () const noexcept -> VectorT { return (max - min) / 2; }
      constexpr void setCenter(const VectorT& v) noexcept { auto r = getRange() ; min = v - r; max = v + r; }
      constexpr void setRange (const VectorT& v) noexcept { auto c = getCenter(); min = c - r; max = c + r; }

      VectorT min;
      VectorT max;
    };

    using AABB2 = AABBT<Vec2>;
    using AABB3 = AABBT<Vec3>;
    using AABB4 = AABBT<Vec4>;
    using AABB  = AABBT<Vec3>;

#if defined(__cplusplus)
  }
}
#endif
