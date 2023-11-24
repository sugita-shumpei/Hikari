#ifndef HK_SHAPE__H
#define HK_SHAPE__H
#if !defined(__CUDACC__)

#include <hikari/object.h>
#include <hikari/object_array.h>
#include <hikari/shape_utils.h>
#include <hikari/math/aabb.h>

#define HK_OBJECT_TYPEID_Shape      HK_UUID_DEFINE(0x4cc662fc, 0xad12, 0x4ba6, 0xa2, 0x93, 0xc7, 0xa9, 0x56, 0x75, 0xfa, 0xe2)
#define HK_OBJECT_TYPEID_ArrayShape HK_UUID_DEFINE(0xa51f3e36, 0x59a3, 0x4c5c, 0x84, 0x2f, 0x7e, 0xa5, 0xb3, 0x6a, 0xa7, 0x40)

#if defined(__cplusplus)
struct HKShape : public HKUnknown
{
    static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Shape; }
    virtual HKAabb HK_API getAabb() const = 0;
};
#else
typedef struct HKShape HKShape;
#endif

HK_NAMESPACE_TYPE_ALIAS(Shape);

HK_EXTERN_C HK_DLL_FUNCTION HKCAabb HK_DLL_FUNCTION_NAME(HKShape_getAabb)(const HKShape *shape);
HK_SHAPE_ARRAY_DEFINE(Shape);

#endif
#endif
