#pragma once
#include <hikari/core/data_type.h>
#include <glm/glm.hpp>
#include <cstdint>
namespace hikari {
  inline namespace core {
    using I8   = int8_t;
    using I16  = int16_t;
    using I32  = int32_t;
    using I64  = int64_t;

    using U8   = uint8_t;
    using U16  = uint16_t;
    using U32  = uint32_t;
    using U64  = uint64_t;

    using F32  = float;
    using F64  = double;

    using Bool = bool;
    using Char = char;

    using Vec2 = glm::vec2;
    using Vec3 = glm::vec3;
    using Vec4 = glm::vec4;

    using Mat2 = glm::mat2;
    using Mat3 = glm::mat3;
    using Mat4 = glm::mat4;

    using Mat2x2 = glm::mat2x2;
    using Mat3x3 = glm::mat3x3;
    using Mat4x4 = glm::mat4x4;

    using Quat   = glm::quat;
  }
}
