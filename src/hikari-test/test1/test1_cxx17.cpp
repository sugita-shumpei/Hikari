#define  HK_RUNTIME_LOAD
#define  HK_MATH_USE_STD_CXX
#define  VK_NO_PROTOTYPES
#include <test1_config.h>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <utility>
#include <thread>
#include <hikari/dynamic_loader.h>
#include <hikari/plugin.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/plugin.h>
#include <hikari/shape/obj_mesh.h>
#include <hikari/graphics/plugin.h>
#include <hikari/graphics/instance.h>
#include <hikari/graphics/vulkan/plugin.h>
#include <hikari/graphics/vulkan/entry.h>
#include <hikari/graphics/opengl/plugin.h>
#include <hikari/graphics/opengl/context.h>
int main()
{
	// Hikariñ{ëÃÇÃPluginì«Ç›çûÇ›Ç…ÇÃÇ›égÇ§Ç±Ç∆
	HKDynamicLoader           loader_core(HK_BUILD_ROOT R"(\src\hikari\core\Debug\hikari-core.dll)");
	HKRefPtr<HKPluginManager> manager  = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(loader_core, HKPluginManager_create)();
	if (manager->load(HK_BUILD_ROOT R"(\src\hikari\shape\Debug\hikari-shape.dll)")) {
		HKRefPtr<HKObjMesh>   obj_mesh = manager->createObjectFromPlugin<HKObjMesh>(HK_OBJECT_TYPEID_PluginShape);
	}
	if (manager->load(HK_BUILD_ROOT R"(\src\hikari\graphics\common\Debug\hikari-graphics-common.dll)")) {
		printf("hikari-graphics-common.dll load!\n");
		HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(manager, (HK_OBJECT_TYPEID_PluginGraphicsCommon), HKGraphicsInstance_getApiName);
		HKCStr res = HKGraphicsInstance_getApiName(nullptr);
	}
	if (manager->load(HK_BUILD_ROOT R"(\src\hikari\graphics\vulkan\Debug\hikari-graphics-vulkan.dll)")) {
		HKRefPtr<HKGraphicsVulkanEntry> entry = manager->createObject<HKGraphicsVulkanEntry>();
		printf("hikari-graphics-vulkan.dll load!\n");
		auto version         = entry->getVersion();
		auto version_variant = HK_GRAPHICS_VULKAN_VERSION_VARIANT(version);
		auto version_major   = HK_GRAPHICS_VULKAN_VERSION_MAJOR(version);
		auto version_minor   = HK_GRAPHICS_VULKAN_VERSION_MINOR(version);
		auto version_patch   = HK_GRAPHICS_VULKAN_VERSION_PATCH(version);
		printf("Vulkan %d.%d.%d.%d\n", version_variant, version_major, version_minor, version_patch);
		{
			auto layer_count = entry->getLayerCount();
			for (HKU32 idx = 0; idx < layer_count; ++idx) {
				printf("Vulkan Layer Name[%d] \"%s\"\n", idx, entry->getLayerName(idx));
			}
		}
		{
			auto extension_count = entry->getExtensionCount();
			for (HKU32 idx = 0; idx < extension_count; ++idx) {
				printf("Vulkan Extension Name[%d] \"%s\"\n", idx, entry->getExtensionName(idx));
			}
		}
	}
	if (manager->load(HK_BUILD_ROOT R"(\src\hikari\graphics\opengl\Debug\hikari-graphics-opengl.dll)")) {
		printf("hikari-graphics-opengl.dll load!\n");
		HKRefPtr<HKGraphicsOpenGLContextManager> context_manager = manager->createObject<HKGraphicsOpenGLContextManager>();
		printf("threadid: %lld\n", context_manager->getThreadID());
		printf("threadid: %lld\n", std::hash<std::thread::id>()(std::this_thread::get_id()));

		std::thread th([&manager]() {
			HKRefPtr<HKGraphicsOpenGLContextManager> context_manager2 = manager->createObject<HKGraphicsOpenGLContextManager>();
			printf("threadid: %lld\n", context_manager2->getThreadID());
			printf("threadid: %lld\n", std::hash<std::thread::id>()(std::this_thread::get_id()));
		});
		th.join();
	}

	HKRefPtr<HKArrayVec3>     arr_vec3 = manager->createObjectFromPlugin<HKArrayVec3>(HK_OBJECT_TYPEID_PluginCore);
	return 0;
}
