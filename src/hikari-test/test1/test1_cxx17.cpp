#define  HK_RUNTIME_LOAD
#define  HK_MATH_USE_STD_CXX

#include <cmath>
#include <hikari/dynamic_loader.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/obj_mesh.h>

int main()
{
	// Hikariñ{ëÃÇÃPluginì«Ç›çûÇ›Ç…ÇÃÇ›égÇ§Ç±Ç∆
	HKDynamicLoader     loader_core(R"(D:\Users\shumpei\Document\CMake\Hikari\build\src\hikari\Debug\hikari.dll)");
	auto plugin_core  = HKRefPtr<HKPlugin>(loader_core.getPlugin());
	auto arr_1        = HKRefPtr<HKArrayVec3>(plugin_core->createObject<HKArrayVec3>());
	auto arr_2        = HKRefPtr<HKArrayVec3>(plugin_core->createObject<HKArrayVec3>());
	return 0;
}

