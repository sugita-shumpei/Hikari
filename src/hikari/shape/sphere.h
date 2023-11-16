#ifndef HK_SHAPE_SPHERE__H
#define HK_SHAPE_SPHERE__H
#include "../shape.h"
#include "../object_array.h"
#define HK_OBJECT_TYPEID_Sphere      HK_UUID_DEFINE(0x3940a449, 0xda93, 0x47dc, 0x9f, 0xa1, 0x43, 0x69, 0x5, 0xaf, 0xc1, 0x8c)
#define HK_OBJECT_TYPEID_ArraySphere HK_UUID_DEFINE(0xea108d25, 0x497b, 0x4468, 0xb0, 0x3e, 0xe6, 0xd6, 0x74, 0xf, 0xbd, 0x3d)
#define HK_OBJECT_SAFE_CAST_Sphere      HK_OBJECT_SAFE_CAST_Shape || iid == HK_OBJECT_TYPEID_Sphere
#define HK_OBJECT_SAFE_CAST_ArraySphere HK_OBJECT_SAFE_CAST_Shape || iid == HK_OBJECT_TYPEID_ArraySphere

#if defined(__cplusplus)
struct HKSphere : public HKShape {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Sphere; }
	static HK_INLINE HKSphere* create();
	static HK_INLINE HKSphere* create(const HKVec3& c, HKF32 r);
	virtual HKSphere* HK_API clone()const = 0;
	virtual void      HK_API setCenter(const HKVec3& c) = 0;
	virtual HKVec3    HK_API getCenter() const = 0;
	virtual void      HK_API setRadius(HKF32 r)= 0;
	virtual HKF32     HK_API getRadius() const = 0;
	HK_INLINE HKSphere* cloneWithRef()const { HKSphere* ptr = clone(); ptr->addRef(); return ptr; }
};

#else
typedef struct HKSphere HKSphere;
#endif
HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create ();
HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_clone(const HKSphere* sphere);
HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create2(HKCVec3 c, HKF32 r);
HK_EXTERN_C HK_DLL void      HK_API HKSphere_setCenter(HKSphere* sp, HKCVec3  c);
HK_EXTERN_C HK_DLL HKCVec3   HK_API HKSphere_getCenter(const HKSphere*       sp);
HK_EXTERN_C HK_DLL void      HK_API HKSphere_setRadius(HKSphere* sp, HKF32    r);
HK_EXTERN_C HK_DLL HKF32     HK_API HKSphere_getRadius(const HKSphere*       sp);
#if defined(__cplusplus)
HK_INLINE HKSphere* HKSphere::create()                         { return HKSphere_create(); }
HK_INLINE HKSphere* HKSphere::create(const HKVec3& c, HKF32 r) { return HKSphere_create2(HKCVec3(c),r); }
#endif
HK_SHAPE_ARRAY_DEFINE(Sphere);
#endif
