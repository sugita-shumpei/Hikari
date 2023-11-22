#ifndef HK_SHAPE_SPHERE__H
#define HK_SHAPE_SPHERE__H
#include "../shape.h"
#include "../object_array.h"
#define HK_OBJECT_TYPEID_Sphere      HK_UUID_DEFINE(0x3940a449, 0xda93, 0x47dc, 0x9f, 0xa1, 0x43, 0x69, 0x5, 0xaf, 0xc1, 0x8c)
#define HK_OBJECT_TYPEID_ArraySphere HK_UUID_DEFINE(0xea108d25, 0x497b, 0x4468, 0xb0, 0x3e, 0xe6, 0xd6, 0x74, 0xf, 0xbd, 0x3d)

#define HK_SPHERE_C_DERIVE_METHODS(TYPE) \
HK_SHAPE_ARRAY_DEFINE(TYPE); \
HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(TYPE,HKSphere,clone,HKSphere*); \
HK_OBJECT_C_DERIVE_METHOD_DECL_2_VOID(TYPE,HKSphere,setCenter,HKCVec3,c); \
HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(TYPE,HKSphere,getCenter,HKVec3); \
HK_OBJECT_C_DERIVE_METHOD_DECL_2_VOID(TYPE,HKSphere,setRadius,HKF32,r); \
HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(TYPE,HKSphere,getRadius,HKF32)

#if defined(__cplusplus)
struct HKSphere : public HKShape {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Sphere; }
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
HK_NAMESPACE_TYPE_ALIAS(Sphere);

HK_EXTERN_C HK_DLL_FUNCTION HKSphere* HK_DLL_FUNCTION_NAME(HKSphere_create   )();
HK_EXTERN_C HK_DLL_FUNCTION HKSphere* HK_DLL_FUNCTION_NAME(HKSphere_clone    )(const HKSphere* sphere);
HK_EXTERN_C HK_DLL_FUNCTION HKSphere* HK_DLL_FUNCTION_NAME(HKSphere_create2  )(HKCVec3 c, HKF32 r);
HK_EXTERN_C HK_DLL_FUNCTION void      HK_DLL_FUNCTION_NAME(HKSphere_setCenter)(HKSphere* sp, HKCVec3  c);
HK_EXTERN_C HK_DLL_FUNCTION HKCVec3   HK_DLL_FUNCTION_NAME(HKSphere_getCenter)(const HKSphere*       sp);
HK_EXTERN_C HK_DLL_FUNCTION void      HK_DLL_FUNCTION_NAME(HKSphere_setRadius)(HKSphere* sp, HKF32    r);
HK_EXTERN_C HK_DLL_FUNCTION HKF32     HK_DLL_FUNCTION_NAME(HKSphere_getRadius)(const HKSphere*       sp);
HK_SHAPE_C_DERIVE_METHODS(HKSphere);

#if defined(__cplusplus)
HK_OBJECT_CREATE_TRAITS(HKSphere);
#endif

HK_SHAPE_ARRAY_DEFINE(Sphere);

#endif
