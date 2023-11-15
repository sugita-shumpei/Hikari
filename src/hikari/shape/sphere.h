#ifndef HK_SHAPE_SPHERE__H
#define HK_SHAPE_SPHERE__H
#include "../shape.h"
#define HK_OBJECT_TYPEID_Sphere HK_UUID_DEFINE(0x3940a449, 0xda93, 0x47dc, 0x9f, 0xa1, 0x43, 0x69, 0x5, 0xaf, 0xc1, 0x8c)
#if defined(__cplusplus)
struct HKSphere : public HKShape {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Sphere; }
	static HK_INLINE HKSphere* create();
	static HK_INLINE HKSphere* create(const HKVec3& c, HKF32 r);
	virtual void   HK_API setCenter(const HKVec3& c) = 0;
	virtual HKVec3 HK_API getCenter() const = 0;
	virtual void   HK_API setRadius(HKF32 r)= 0;
	virtual HKF32  HK_API getRadius() const = 0;
};
#else
typedef struct HKSphere HKSphere;
#endif
HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create ();
HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create2(HKCVec3 c, HKF32 r);
HK_EXTERN_C HK_DLL void      HK_API HKSphere_setCenter(HKSphere* sp, HKCVec3  c);
HK_EXTERN_C HK_DLL HKCVec3   HK_API HKSphere_getCenter(const HKSphere*       sp);
HK_EXTERN_C HK_DLL void      HK_API HKSphere_setRadius(HKSphere* sp, HKF32    r);
HK_EXTERN_C HK_DLL HKF32     HK_API HKSphere_getRadius(const HKSphere*       sp);
#if defined(__cplusplus)
HK_INLINE HKSphere* HKSphere::create()                         { return HKSphere_create(); }
HK_INLINE HKSphere* HKSphere::create(const HKVec3& c, HKF32 r) { return HKSphere_create2(HKCVec3(c),r); }
#endif
#endif
