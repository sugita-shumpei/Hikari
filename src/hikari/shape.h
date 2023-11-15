#ifndef HK_SHAPE__H
#define HK_SHAPE__H

#include "object.h"
#include "math/aabb.h"

#define HK_OBJECT_TYPEID_Shape HK_UUID_DEFINE(0x4cc662fc, 0xad12, 0x4ba6, 0xa2, 0x93, 0xc7, 0xa9, 0x56, 0x75, 0xfa, 0xe2)

#if defined(__cplusplus)
struct HKShape : public HKUnknown {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Shape; }
	virtual HKAabb HK_API getAabb() const = 0;
};
#else
typedef struct HKShape HKShape;
#endif

HK_EXTERN_C HK_DLL HKCAabb HK_API HKShape_getAabb(const HKShape* shape);

#endif
