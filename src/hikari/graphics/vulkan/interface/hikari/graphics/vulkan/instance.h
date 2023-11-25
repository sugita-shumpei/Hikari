#ifndef HK_GRAPHICS_VULKAN_INSTANCE__H
#define HK_GRAPHICS_VULKAN_INSTANCE__H

#if !defined(__CUDACC__)
#include <hikari/graphics/instance.h>
#define HK_OBJECT_TYPEID_GraphicsVulkanInstance HK_UUID_DEFINE(0x10594ced, 0x5c35, 0x4d9a, 0xa4, 0xfb, 0x72, 0x34, 0xff, 0xa7, 0x5d, 0xd2)

#if defined(__cplusplus)
struct HKGraphicsVulkanInstance : public HKGraphicsInstance {
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsVulkanInstance; }
};
#else
typedef struct HKGraphicsInstance HKGraphicsInstance;
#endif
HK_NAMESPACE_TYPE_ALIAS(GraphicsVulkanInstance);

#endif
#endif
