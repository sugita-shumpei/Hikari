#define  HK_MATH_USE_STD_CXX

#include <cmath>
#include <test1_config.h>
#include <hikari/dynamic_loader.h>
#include <hikari/plugin.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/plugin.h>
#include <hikari/shape/obj_mesh.h>

int main()
{
	// Hikariñ{ëÃÇÃPluginì«Ç›çûÇ›Ç…ÇÃÇ›égÇ§Ç±Ç∆
	HKRefPtr<HKPluginManager> manager = HKRefPtr<HKPluginManager>::create();
	if (manager->load(HK_BUILD_ROOT R"(\src\hikari\shape\Debug\hikari-shape.dll)")) {
		HKRefPtr<HKObjMesh> obj_mesh = manager->createObjectFromPlugin<HKObjMesh>(HK_OBJECT_TYPEID_PluginShape);
	}
	HKRefPtr<HKArrayVec3>   arr_vec3 = manager->createObjectFromPlugin<HKArrayVec3>(HK_OBJECT_TYPEID_PluginCore);
	return 0;
}
