#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/transform.hpp>
#include <hikari/core/types/data_type.h>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    typedef glm::mat2 Mat2;
    typedef glm::mat3 Mat3;
    typedef glm::mat4 Mat4;
    typedef glm::mat4 Matrix;
    using glm::determinant;
    using glm::inverse;
    using glm::transpose;
    using glm::inverseTranspose;
    using glm::translate;
    using glm::scale;
    using glm::frustum;
    using glm::lookAt;
    using glm::perspective;
    using glm::perspectiveFov;

    typedef Array<Mat2> ArrayMat2;
    typedef Array<Mat3> ArrayMat3;
    typedef Array<Mat4> ArrayMat4;
    typedef Array<Mat4> ArrayMatrix;

#if defined(__cplusplus)
  }
}
#endif
