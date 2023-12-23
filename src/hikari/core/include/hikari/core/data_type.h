#pragma once
#include <Imath/half.h>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stduuid/uuid.h>
#include <cstdint>
#include <string>
namespace hikari {
  using I8  = int8_t ;
  using I16 = int16_t;
  using I32 = int32_t;
  using I64 = int64_t;

  using U8  = uint8_t;
  using U16 = uint16_t;
  using U32 = uint32_t;
  using U64 = uint64_t;

  using F16  = Imath::half;
  using F32  = float;
  using F64  = double;

  using Byte = std::byte;
  using Bool = bool;
  using Char = char;

  using CStr   = const char*;
  using String = std::string;

  using Vec2   = glm::vec2;
  using Vec3   = glm::vec3;
  using Vec4   = glm::vec4;

  using Mat2x2 = glm::mat2x2;
  using Mat2x3 = glm::mat2x3;
  using Mat2x4 = glm::mat2x4;
  using Mat3x2 = glm::mat3x2;
  using Mat3x3 = glm::mat3x3;
  using Mat3x4 = glm::mat3x4;
  using Mat4x2 = glm::mat4x2;
  using Mat4x3 = glm::mat4x3;
  using Mat4x4 = glm::mat4x4;

  using Quat   = glm::quat;
  using Uuid   = uuids::uuid;
}
