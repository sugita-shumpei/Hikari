#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <glm/glm.hpp>
#include <hikari/core/types/data_type.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    typedef glm::vec2 Vec2;
    typedef glm::vec3 Vec3;
    typedef glm::vec4 Vec4;
    typedef glm::vec4 Vector;
    using   glm::cross;
    using   glm::length;
    using   glm::dot;
    using   glm::normalize;
    using   glm::radians;
    using   glm::degrees;
    typedef Array<Vec2> ArrayVec2;
    typedef Array<Vec3> ArrayVec3;
    typedef Array<Vec4> ArrayVec4;
    typedef Array<Vec4> ArrayVector;

#if defined(__cplusplus)
  }
}
#endif
