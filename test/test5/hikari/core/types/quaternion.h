#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <hikari/core/types/data_type.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    typedef glm::quat Quat;
    typedef Quat Quaternion;
    using   glm::quat_cast;
    using   glm::toMat3;
    using   glm::toMat4;
    using   glm::toQuat;
    typedef Array<Quat> ArrayQuat;
    inline auto fromEuler(const Vec3& v) noexcept -> Quat {
      return Quat(radians(v));
    }
    inline auto toEuler(const Quat& v) noexcept -> Vec3 {
      return degrees(glm::eulerAngles(v));
    }


#if defined(__cplusplus)
  }
}
#endif
