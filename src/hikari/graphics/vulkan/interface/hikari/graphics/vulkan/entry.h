#ifndef HK_GRAPHICS_VULKAN_ENTRY__H
#define HK_GRAPHICS_VULKAN_ENTRY__H
#if !defined(__CUDACC__)
#include <hikari/object.h>
#include <hikari/uuid.h>
#include <hikari/graphics/entry.h>
#include <hikari/graphics/vulkan/common.h>

#define HK_OBJECT_TYPEID_GraphicsVulkanEntry HK_UUID_DEFINE(0x1943324b, 0x5720, 0x48e1, 0xa9, 0xfb, 0xc1, 0xc, 0x3f, 0xd2, 0x5b, 0x9)

#if  defined(__cplusplus)
struct HKGraphicsVulkanEntry : public HKGraphicsEntry
{
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsVulkanEntry; }

	virtual Pfn_HKGraphicsVulkan_VoidFunction HK_API getProcAddress(HKCStr name)      const = 0;
	virtual HKU32                             HK_API getVersion()                     const = 0;
	virtual HKBool                            HK_API hasExtensionName(HKCStr name)    const = 0;
	virtual HKCStr                            HK_API getExtensionName(HKU32 idx)      const = 0;
	virtual HKU32                             HK_API getExtensionCount()              const = 0;
	virtual HKBool                            HK_API hasLayerName(HKCStr name)        const = 0;
	virtual HKCStr                            HK_API getLayerName(HKU32 idx)          const = 0;
	virtual HKU32                             HK_API getLayerCount()                  const = 0;

	template<typename FunctionPtr>
	HK_INLINE FunctionPtr getProcAddress(const char* name)const {
		return reinterpret_cast<FunctionPtr>(getProcAddress(name));
	}
};
#else
typedef struct HKGraphicsVulkanEntry HKGraphicsVulkanEntry;
#endif
HK_EXTERN_C HK_DLL_FUNCTION Pfn_HKGraphicsVulkan_VoidFunction HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getProcAddress)(const HKGraphicsVulkanEntry* entry, HKCStr name);
HK_EXTERN_C HK_DLL_FUNCTION HKU32                             HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getVersion    )(const HKGraphicsVulkanEntry* entry);
HK_EXTERN_C HK_DLL_FUNCTION HKBool                            HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_hasExtensionName)(const HKGraphicsVulkanEntry* entry, HKCStr name);
HK_EXTERN_C HK_DLL_FUNCTION HKCStr                            HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getExtensionName)(const HKGraphicsVulkanEntry* entry, HKU32 idx);
HK_EXTERN_C HK_DLL_FUNCTION HKU32                             HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getExtensionCount)(const HKGraphicsVulkanEntry* entry);
HK_EXTERN_C HK_DLL_FUNCTION HKBool                            HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_hasLayerName)(const HKGraphicsVulkanEntry* entry, HKCStr name);
HK_EXTERN_C HK_DLL_FUNCTION HKCStr                            HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getLayerName)(const HKGraphicsVulkanEntry* entry, HKU32 idx);
HK_EXTERN_C HK_DLL_FUNCTION HKU32                             HK_DLL_FUNCTION_NAME(HKGraphicsVulkanEntry_getLayerCount)(const HKGraphicsVulkanEntry* entry);

#endif
#endif