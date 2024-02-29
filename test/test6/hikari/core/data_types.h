#pragma once
#include <hikari/core/platform.h>

#if defined(HK_LANG_CUDA_CXX)
#include <cuda.h>
#endif

#if defined(HK_LANG_CXX_HOST)
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <imath/half.h>
#include <string>
#endif


#if defined(HK_LANG_CXX) 
namespace hikari {
#endif
  // GLSL
#if defined(HK_LANG_GLSL)
#define Bool bool
#define I32  int
#define U32  uint
#define F32  float
#define F64  double
#define UVec2 uvec2
#define UVec3 uvec3
#define UVec4 uvec4
#define IVec2 ivec2
#define IVec3 ivec3
#define IVec4 ivec4
#define Vec2  vec2
#define Vec3  vec3
#define Vec4  vec4
#define Mat2  mat2;
#define Mat3  mat3;
#define Mat4  mat4;
#define Mat2x2 mat2x2
#define Mat3x2 mat3x2
#define Mat4x2 mat4x2
#define Mat2x3 mat2x3
#define Mat3x3 mat3x3
#define Mat4x3 mat4x3
#define Mat2x4 mat2x4
#define Mat3x4 mat3x4
#define Mat4x4 mat4x4
#define Quat   vec4
#endif
  // HLSL
#if defined(HL_LANG_HLSL)
  typedef bool     Bool;
  typedef int      I32;
  typedef uint     U32;
  typedef half     F16;
  typedef float    F32;
  typedef double   F64;
  typedef int2     IVec2;
  typedef int3     IVec3;
  typedef int4     IVec4;
  typedef uint2    UVec2;
  typedef uint3    UVec3;
  typedef uint4    UVec4;
  typedef float2   Vec2;
  typedef float3   Vec3;
  typedef float4   Vec4;
  typedef float2x2 Mat2x2;
  typedef float3x3 Mat3x3;
  typedef float4x4 Mat4x4;
  typedef float2x2 Mat2x2;
  typedef float3x2 Mat3x2;
  typedef float4x2 Mat4x2;
  typedef float2x3 Mat2x3;
  typedef float3x3 Mat3x3;
  typedef float4x3 Mat4x3;
  typedef float2x4 Mat2x4;
  typedef float3x4 Mat3x4;
  typedef float4x4 Mat4x4;
  // C/C++
#endif
  // C++
#if defined(HK_LANG_CXX)
  typedef signed char        I8;
  typedef signed short       I16;
  typedef signed int         I32;
  typedef signed long long   I64;
  typedef unsigned char      U8;
  typedef unsigned short     U16;
  typedef unsigned int       U32;
  typedef unsigned long long U64;
  typedef float              F32;
  typedef double             F64;
  typedef bool               Bool;
  typedef char               Char;
  // CUDA
#if defined(HK_LANG_CUDA_CXX)
  using  Vec2 = float2;
  using  Vec3 = float3;
  using  Vec4 = float4;
  using  Vec2 = int2;
  using  Vec3 = int3;
  using  Vec4 = int4;
  using  UVec2 = uint2;
  using  UVec3 = uint3;
  using  UVec4 = uint4;
  struct Mat2x2 { Vec2 data[2]; };
  struct Mat3x2 { Vec3 data[2]; };
  struct Mat4x2 { Vec4 data[2]; };
  struct Mat2x3 { Vec2 data[3]; };
  struct Mat3x3 { Vec3 data[3]; };
  struct Mat4x3 { Vec4 data[3]; };
  struct Mat2x4 { Vec2 data[4]; };
  struct Mat3x4 { Vec3 data[4]; };
  struct Mat4x4 { Vec4 data[4]; };
  using  Mat2 = Mat2x2;
  using  Mat3 = Mat3x3;
  using  Mat4 = Mat4x4;
  using  Quat = float4;
  using  F16  = __half;
#else
  // C++ Host
  using Vec2   = glm::vec2;
  using Vec3   = glm::vec3;
  using Vec4   = glm::vec4;
  using UVec2  = glm::uvec2;
  using UVec3  = glm::uvec3;
  using UVec4  = glm::uvec4;
  using IVec2  = glm::ivec2;
  using IVec3  = glm::ivec3;
  using IVec4  = glm::ivec4;
  using DVec2  = glm::dvec2;
  using DVec3  = glm::dvec3;
  using DVec4  = glm::dvec4;
  using Mat2   = glm::mat2;
  using Mat3   = glm::mat3;
  using Mat4   = glm::mat4;
  using Mat2x2 = glm::mat2x2;
  using Mat3x2 = glm::mat3x2;
  using Mat4x2 = glm::mat4x2;
  using Mat2x3 = glm::mat2x3;
  using Mat3x3 = glm::mat3x3;
  using Mat4x3 = glm::mat4x3;
  using Mat2x4 = glm::mat2x4;
  using Mat3x4 = glm::mat3x4;
  using Mat4x4 = glm::mat4x4;
  using Quat   = glm::vec4;
  using F16    = Imath::half;
  using CStr   = const Char*;
  using String = std::string;
#endif

#endif

#if defined(HK_LANG_CXX) 
}
#endif
