#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/math/quat.h>
#include <hikari/math/aabb.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/obj_mesh.h>
#include <stdio.h>
#include <assert.h>
int main() {
	HKObjMesh* obj_mesh           = HKObjMesh_create();
	HKBool res                    = HKObjMesh_loadFile(obj_mesh, "D:\\Users\\shumpei\\Document\\Github\\RTLib\\Data\\Models\\Sponza\\sponza.obj");
	if (res){
		HKArrayVec3   * vertices  = HKObjMesh_getVertices(obj_mesh);
		HKArrayVec3   * normals   = HKObjMesh_getNormals (obj_mesh);
		HKArrayVec4   * tangents  = HKObjMesh_getTangents(obj_mesh);
		HKArrayVec2   * uvs       = HKObjMesh_getUVs(obj_mesh,0 );
		HKArraySubMesh* submeshes = HKObjMesh_getSubMeshes(obj_mesh);

		for (HKU32 i = 0; i < HKArraySubMesh_getCount(submeshes); ++i) {
			printf("submeshes: %s\n", HKObjSubMesh_getName((const HKObjSubMesh*)HKArraySubMesh_internal_getPointer_const(submeshes)[i]));
		}

		HKArraySubMesh_release(submeshes);
		HKArrayVec3_release(vertices);
		HKArrayVec3_release(normals);
		HKArrayVec4_release(tangents);
		HKArrayVec2_release(uvs);
	}
	HKObjMesh_release(obj_mesh);
	return 0;
}
