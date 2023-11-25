#ifndef HK_GRAPHICS_ENTRY__H
#define HK_GRAPHICS_ENTRY__H
#if !defined(__CUDACC__)
#include <hikari/object.h>
#define HK_OBJECT_TYPEID_GraphicsEntry HK_UUID_DEFINE(0xd2c8ee8d, 0xe3dd, 0x428a, 0xb7, 0x54, 0x11, 0xe7, 0xb8, 0x56, 0x70, 0xdd)
#if defined(__cplusplus)
struct HKGraphicsEntry : public HKUnknown 
{
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsEntry;  }

};
#else
typedef struct HKGraphicsEntry HKGraphicsEntry;
#endif
HK_EXTERN_C typedef HKGraphicsEntry* (HK_API* Pfn_HKGraphicsEntry_create)();

#endif
#endif
