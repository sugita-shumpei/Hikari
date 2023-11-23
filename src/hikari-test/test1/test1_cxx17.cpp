#define  HK_RUNTIME_LOAD
#define  HK_MATH_USE_STD_CXX

#include <cmath>
#include <hikari/dynamic_loader.h>
#include <hikari/plugin.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/plugin.h>
#include <hikari/shape/obj_mesh.h>

int main()
{
	// Hikari�{�̂�Plugin�ǂݍ��݂ɂ̂ݎg������
	HKDynamicLoader           loader_core(R"(D:\Users\shumpei\Document\CMake\Hikari\build\src\hikari\Debug\hikari.dll)");
	HKRefPtr<HKPluginManager> manager  = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(loader_core, HKPluginManager_create)();
	if (manager->load(R"(D:\Users\shumpei\Document\CMake\Hikari\build\src\hikari\Debug\hikari-shape.dll)")) {
		HKRefPtr<HKObjMesh>   obj_mesh = manager->createObjectFromPlugin<HKObjMesh>(HK_OBJECT_TYPEID_PluginShape);
	}
	HKRefPtr<HKArrayVec3>     arr_vec3 = manager->createObjectFromPlugin<HKArrayVec3>(HK_OBJECT_TYPEID_PluginCore);
	return 0;
}
