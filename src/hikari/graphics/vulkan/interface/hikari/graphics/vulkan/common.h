#ifndef HK_GRAPHICS_VULKAN_COMMON__H
#define HK_GRAPHICS_VULKAN_COMMON__H
#ifndef __CUDACC__

#include <hikari/platform.h>
#include <hikari/data_type.h>
#define HK_GRAPHICS_VULKAN_MAKE_VERSION (variant,major,minor,patch) \
    ((((HKU32)(variant)) << 29U) | (((HKU32)(major)) << 22U) | (((HKU32)(minor)) << 12U) | ((HKU32)(patch)))

#define HK_GRAPHICS_VULKAN_VERSION_MAJOR(VERSION)   ((HKU32)(version) >> 22U)
#define HK_GRAPHICS_VULKAN_VERSION_MINOR(VERSION)   (((HKU32)(version) >> 12U) & 0x3FFU)
#define HK_GRAPHICS_VULKAN_VERSION_PATCH(VERSION)   ((HKU32)(version) & 0xFFFU)
#define HK_GRAPHICS_VULKAN_VERSION_VARIANT(VERSION) ((HKU32)(version) >> 29U)
#define HK_GRAPHICS_VULKAN_MAX_EXTENSION_NAME_SIZE 256U
#define HK_GRAPHICS_VULKAN_MAX_DESCRIPTION_SIZE    256U

typedef void (HK_API* Pfn_HKGraphicsVulkan_VoidFunction)();

typedef struct HKGraphicsVulkanExtensionProperties {
    HKChar extensionName[HK_GRAPHICS_VULKAN_MAX_EXTENSION_NAME_SIZE];
    HKU32  specVersion;
} HKGraphicsVulkanExtensionProperties;

typedef struct HKGraphicsVulkanLayerProperties {
    HKChar   layerName[HK_GRAPHICS_VULKAN_MAX_EXTENSION_NAME_SIZE];
    HKU32    specVersion;
    HKU32    implementationVersion;
    HKChar   description[HK_GRAPHICS_VULKAN_MAX_DESCRIPTION_SIZE];
} HKGraphicsVulkanLayerProperties;

#endif
#endif
