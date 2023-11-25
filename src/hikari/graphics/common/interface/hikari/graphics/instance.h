#ifndef HK_GRAPHICS_INSTANCE__H
#define HK_GRAPHICS_INSTANCE__H

#if !defined(__CUDACC__)
#include <hikari/object.h>
#define HK_OBJECT_TYPEID_GraphicsInstance HK_UUID_DEFINE(0x2e1dd55d, 0xe189, 0x403b, 0xb7, 0x8c, 0x73, 0xad, 0x20, 0x7a, 0xbc, 0xa7)
#if defined(__cplusplus)
struct HKGraphicsInstance : public HKUnknown {
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsInstance;  }
	virtual HKCStr HK_API getApiName() const = 0;
};
#else
typedef struct HKGraphicsInstance HKGraphicsInstance;
#endif
HK_NAMESPACE_TYPE_ALIAS(GraphicsInstance);
HK_EXTERN_C HK_DLL_FUNCTION HKCStr HK_DLL_FUNCTION_NAME(HKGraphicsInstance_getApiName)(const HKGraphicsInstance*);

#endif
#endif
