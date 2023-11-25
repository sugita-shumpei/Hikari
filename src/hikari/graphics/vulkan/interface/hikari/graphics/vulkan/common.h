#ifndef HK_GRAPHICS_VULKAN_COMMON__H
#define HK_GRAPHICS_VULKAN_COMMON__H
#ifndef __CUDACC__

#include <hikari/platform.h>
#include <hikari/data_type.h>

#define HK_GRAPHICS_VULKAN_VERSION_MAJOR(VERSION)   ((HKU32)(version) >> 22U)
#define HK_GRAPHICS_VULKAN_VERSION_MINOR(VERSION)   (((HKU32)(version) >> 12U) & 0x3FFU)
#define HK_GRAPHICS_VULKAN_VERSION_PATCH(VERSION)   ((HKU32)(version) & 0xFFFU)
#define HK_GRAPHICS_VULKAN_VERSION_VARIANT(VERSION) ((HKU32)(version) >> 29U)
typedef void (HK_API* Pfn_HKGraphicsVulkan_VoidFunction)();


#endif
#endif

