#ifndef HK_VALUE_ARRAY__H
#define HK_VALUE_ARRAY__H
#if !defined(__CUDACC__)

#include <hikari/data_type.h>
#include <hikari/object.h>
#include "value_array_utils.h"
#include "color.h"
#include "math/vec.h"
#include "math/matrix.h"

#define HK_OBJECT_TYPEID_ArrayU8     HK_UUID_DEFINE(0xd2ad77fc, 0x1eec, 0x43e9, 0xb9, 0x79, 0x3b, 0xb9, 0x9, 0x66, 0x86, 0x16)
#define HK_OBJECT_TYPEID_ArrayU16    HK_UUID_DEFINE(0xbb434d7e, 0xdfc, 0x4847, 0xa7, 0x8, 0x93, 0x74, 0x82, 0xad, 0xf0, 0x21)
#define HK_OBJECT_TYPEID_ArrayU32    HK_UUID_DEFINE(0xdd7a7431, 0x6925, 0x4237, 0x85, 0x7c, 0xa6, 0xd, 0x95, 0x69, 0x83, 0xd)
#define HK_OBJECT_TYPEID_ArrayU64    HK_UUID_DEFINE(0x9092207, 0x12e7, 0x41f4, 0xa7, 0xb3, 0x96, 0x2f, 0xd, 0x95, 0x56, 0x2e)

#define HK_OBJECT_TYPEID_ArrayI8     HK_UUID_DEFINE(0xf72d3383, 0xb6c4, 0x4805, 0x99, 0x2e, 0x83, 0x82, 0x5e, 0x34, 0x55, 0x85)
#define HK_OBJECT_TYPEID_ArrayI16    HK_UUID_DEFINE(0x2c69041a, 0x7c2e, 0x4a96, 0xbd, 0xd5, 0x57, 0x4a, 0x95, 0x75, 0xfa, 0x80)
#define HK_OBJECT_TYPEID_ArrayI32    HK_UUID_DEFINE(0xaa7dab1b, 0x8c17, 0x4752, 0x91, 0x90, 0x5d, 0xd7, 0xd6, 0x97, 0x77, 0xc8)
#define HK_OBJECT_TYPEID_ArrayI64    HK_UUID_DEFINE(0x9a17c767, 0xa537, 0x4f7a, 0x82, 0xd1, 0xca, 0xab, 0xc6, 0x1e, 0xde, 0x37)

#define HK_OBJECT_TYPEID_ArrayF32    HK_UUID_DEFINE(0x6d558c26, 0xd2c1, 0x4be6, 0x98, 0xfb, 0x9f, 0x95, 0xdc, 0x78, 0xf3, 0xec)
#define HK_OBJECT_TYPEID_ArrayF64    HK_UUID_DEFINE(0xf1001c50, 0x459, 0x43be, 0x8f, 0x80, 0x64, 0xf6, 0x87, 0x61, 0x21, 0x34)

#define HK_OBJECT_TYPEID_ArrayByte   HK_UUID_DEFINE(0x6c25a056, 0x6279, 0x4e2b, 0x9e, 0xff, 0xaa, 0xec, 0x6f, 0xe4, 0x3c, 0x7b)
#define HK_OBJECT_TYPEID_ArrayBool   HK_UUID_DEFINE(0xbff6655c, 0x8110, 0x4cae, 0x95, 0x11, 0x65, 0x6c, 0x6d, 0xd8, 0x47, 0x67)
#define HK_OBJECT_TYPEID_ArrayChar   HK_UUID_DEFINE(0xc2aee505, 0x47be, 0x4b8a, 0xb9, 0xf2, 0x11, 0xb7, 0xf2, 0x73, 0x94, 0xfd)

#define HK_OBJECT_TYPEID_ArrayVec2   HK_UUID_DEFINE(0xef915076, 0x7249, 0x40c1, 0xbc, 0x2, 0x9a, 0x1e, 0xc8, 0xf6, 0x9a, 0xfd)
#define HK_OBJECT_TYPEID_ArrayVec3   HK_UUID_DEFINE(0x3908ee67, 0x31fe, 0x4cbd, 0x99, 0xd2, 0x10, 0x58, 0x50, 0x15, 0xb0, 0x1)
#define HK_OBJECT_TYPEID_ArrayVec4   HK_UUID_DEFINE(0x4c5b3d98, 0x5e84, 0x49d7, 0x8e, 0xf5, 0xa3, 0xec, 0xfa, 0xba, 0xe7, 0x69)

#define HK_OBJECT_TYPEID_ArrayMat2x2 HK_UUID_DEFINE(0xd6641069, 0xe54d, 0x4fe8, 0xb3, 0xda, 0x79, 0x32, 0x4c, 0x8f, 0x7a, 0x14)
#define HK_OBJECT_TYPEID_ArrayMat3x3 HK_UUID_DEFINE(0xc0f612e6, 0x4727, 0x4731, 0xa7, 0xbe, 0xc2, 0xf6, 0x2c, 0xac, 0xd9, 0x9d)
#define HK_OBJECT_TYPEID_ArrayMat4x4 HK_UUID_DEFINE(0xd148312f, 0x543b, 0x4ab8, 0x9f, 0xae, 0xa9, 0xe0, 0x1d, 0x66, 0x9f, 0x6b)

#define HK_OBJECT_TYPEID_ArrayColor  HK_UUID_DEFINE(0xd31e7a79, 0xf9a0, 0x4d7c, 0xa0, 0xfc, 0xcc, 0xc5, 0x53, 0x59, 0xb9, 0x6)
#define HK_OBJECT_TYPEID_ArrayColor8 HK_UUID_DEFINE(0xdb55e00e, 0xfc63, 0x46d1, 0xb3, 0xe9, 0x7 , 0x87, 0x52, 0x2f, 0x4e, 0x0)

HK_VALUE_ARRAY_DEFINE(U8);
HK_VALUE_ARRAY_DEFINE(U16);
HK_VALUE_ARRAY_DEFINE(U32);
HK_VALUE_ARRAY_DEFINE(U64);

HK_VALUE_ARRAY_DEFINE(I8);
HK_VALUE_ARRAY_DEFINE(I16);
HK_VALUE_ARRAY_DEFINE(I32);
HK_VALUE_ARRAY_DEFINE(I64);

HK_VALUE_ARRAY_DEFINE(F32);
HK_VALUE_ARRAY_DEFINE(F64);

HK_VALUE_ARRAY_DEFINE(Byte);
HK_VALUE_ARRAY_DEFINE(Char);

HK_VALUE_ARRAY_DEFINE(Vec2);
HK_VALUE_ARRAY_DEFINE(Vec3);
HK_VALUE_ARRAY_DEFINE(Vec4);

HK_VALUE_ARRAY_DEFINE(Mat2x2);
HK_VALUE_ARRAY_DEFINE(Mat3x3);
HK_VALUE_ARRAY_DEFINE(Mat4x4);

HK_VALUE_ARRAY_DEFINE(Color);
HK_VALUE_ARRAY_DEFINE(Color8);

#if defined(__cplusplus)
template<>  struct HKArrayTraits<HKU8>     { typedef HKArrayU8  type; };
template<>  struct HKArrayTraits<HKU16>    { typedef HKArrayU16 type; };
template<>  struct HKArrayTraits<HKU32>    { typedef HKArrayU32 type; };
template<>  struct HKArrayTraits<HKU64>    { typedef HKArrayU64 type; };
template<>  struct HKArrayTraits<HKI8>     { typedef HKArrayI8  type; };
template<>  struct HKArrayTraits<HKI16>    { typedef HKArrayI16 type; };
template<>  struct HKArrayTraits<HKI32>    { typedef HKArrayI32 type; };
template<>  struct HKArrayTraits<HKI64>    { typedef HKArrayI64 type; };
template<>  struct HKArrayTraits<HKF32>    { typedef HKArrayF32 type; };
template<>  struct HKArrayTraits<HKF64>    { typedef HKArrayF64 type; };
template<>  struct HKArrayTraits<HKVec2>   { typedef HKArrayVec2 type; };
template<>  struct HKArrayTraits<HKVec3>   { typedef HKArrayVec3 type; };
template<>  struct HKArrayTraits<HKVec4>   { typedef HKArrayVec4 type; };
template<>  struct HKArrayTraits<HKMat2x2> { typedef HKArrayMat2x2 type; };
template<>  struct HKArrayTraits<HKMat3x3> { typedef HKArrayMat3x3 type; };
template<>  struct HKArrayTraits<HKMat4x4> { typedef HKArrayMat4x4 type; };
template<>  struct HKArrayTraits<HKColor > { typedef HKArrayColor  type; };
template<>  struct HKArrayTraits<HKColor8> { typedef HKArrayColor8 type; };
#endif

#endif
#endif