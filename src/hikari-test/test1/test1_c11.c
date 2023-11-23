#define HK_RUNTIME_LOAD
#define HK_MATH_USE_STD_C

#include <stdio.h>
#include <assert.h>

#include <hikari/dynamic_loader.h>
#include <hikari/plugin.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/math/quat.h>
#include <hikari/math/aabb.h>
#include <hikari/shape/plugin.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/obj_mesh.h>

int main()
{
	HKDynamicLoader dl_hikari = HKDynamicLoader_load("D:\\Users\\shumpei\\Document\\CMake\\Hikari\\build\\src\\hikari\\Debug\\hikari.dll");
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKUnknown_addRef);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKUnknown_release);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKUnknown_queryInterface);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_create);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_load);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_unload);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_createObject);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_createObjectFromPlugin);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_internal_getProcAddress);
	HK_DYNAMIC_LOADER_INIT_FUNCTION(dl_hikari, HKPluginManager_internal_getProcAddressFromPlugin);

	HKPluginManager *pl_manager = HKPluginManager_create();
	if (HKPluginManager_load(pl_manager, "D:\\Users\\shumpei\\Document\\CMake\\Hikari\\build\\src\\hikari\\Debug\\hikari-shape.dll"))
	{
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKObjMesh_setFilename);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKObjSubMesh_getName);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKMesh_getSubMeshes);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKMesh_getVertices);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKMesh_getNormals);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKMesh_getUVs);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKArraySubMesh_internal_getPointer);
		 HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(pl_manager, (HK_OBJECT_TYPEID_PluginShape), HKArraySubMesh_getCount);

		 HKObjMesh* obj_mesh   = (HKObjMesh*)HKPluginManager_createObjectFromPlugin(pl_manager, HK_OBJECT_TYPEID_PluginShape, HK_OBJECT_TYPEID_ObjMesh);
		 HKObjMesh_setFilename(obj_mesh, "D:\\Users\\shumpei\\Document\\Github\\RTLib\\Data\\Models\\Sponza\\sponza.obj");

		 HKMesh*    mesh       = NULL;
		 if (HKUnknown_queryInterface((HKUnknown*)obj_mesh, HK_OBJECT_TYPEID_Mesh, (void**)&mesh)) {
			HKArraySubMesh* submeshes   = HKMesh_getSubMeshes(mesh);
			HKArrayVec3*    vertices    = HKMesh_getVertices((HKMesh*)mesh);
			HKArrayVec3*    normals     = HKMesh_getNormals((HKMesh*)mesh);
			HKArrayVec2*    uv0s        = HKMesh_getUVs((HKMesh*)mesh, 0);
			HKSubMesh**     p_submeshes = HKArraySubMesh_internal_getPointer(submeshes);

			for (HKU32 i = 0;i<HKArraySubMesh_getCount(submeshes);++i){
				HKObjSubMesh* obj_submesh = NULL;
				if (HKUnknown_queryInterface((HKUnknown*)p_submeshes[i], HK_OBJECT_TYPEID_ObjSubMesh, (void**)&obj_submesh)) {
					printf("submesh[%d].name=%s\n", i, HKObjSubMesh_getName(obj_submesh));
					HKUnknown_release((HKUnknown*)obj_submesh);
				}
			}

			HKUnknown_release((HKUnknown*)uv0s);
			HKUnknown_release((HKUnknown*)normals);
			HKUnknown_release((HKUnknown*)vertices);
			HKUnknown_release((HKUnknown*)submeshes);
			HKUnknown_release((HKUnknown*)mesh);
		}
		HKUnknown_release((HKUnknown*)obj_mesh);
		HKVec3 unit_x = HKVec3_create3(1.0f, 0.0f, 0.0f);
		HKVec3 unit_y = HKVec3_create3(0.0f, 1.0f, 0.0f);
		HKVec3 unit_z = HKVec3_cross(&unit_x, &unit_y);

		printf("%f %f %f\n", unit_z.x, unit_z.y, unit_z.z);
	}
	HKUnknown_release((HKUnknown *)pl_manager);
	HKDynamicLoader_unload(&dl_hikari);
	return 0;
}
