#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/math/quat.h>
#include <hikari/math/aabb.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/transform_graph.h>
#include <stdio.h>
#include <assert.h>
int main() {
	// ’ÊíŒ^‚ÌOBJECT—p‚ÌRefPtr
	HKMesh* mesh = HKMesh_create();// 0->1->0
	{
		// ”z—ñŒ^‚ÌOBJECT—p‚ÌRefPtr
		HKArrayVec3* vertices = HKArrayVec3_create();
		HKArrayVec3_setCount(vertices,3);
		HKArrayVec3_internal_getPointer(vertices)[0] = HKVec3_create3(-1.0f, -1.0f, 0.0f);
		HKArrayVec3_internal_getPointer(vertices)[1] = HKVec3_create3(+3.0f, -1.0f, 0.0f);
		HKArrayVec3_internal_getPointer(vertices)[2] = HKVec3_create3(-1.0f, +3.0f, 0.0f);

		// ”z—ñŒ^‚ÌOBJECT—p‚ÌRefPtr
		HKArrayVec3* normals = HKArrayVec3_create();
		HKArrayVec3_setCount(normals, 3);
		HKArrayVec3_internal_getPointer(normals)[0] = HKVec3_create3(0.0f, 0.0f, 1.0f);
		HKArrayVec3_internal_getPointer(normals)[1] = HKVec3_create3(0.0f, 0.0f, 1.0f);
		HKArrayVec3_internal_getPointer(normals)[2] = HKVec3_create3(0.0f, 0.0f, 1.0f);

		HKMesh_setVertices(mesh,vertices);
		HKMesh_setNormals(mesh, normals);

		HKUnknown_release((HKUnknown*)vertices);
		HKUnknown_release((HKUnknown*)normals);
	}
	{
		// ”z—ñŒ^‚ÌOBJECT—p‚ÌRefPtr
		HKArrayVec3* vertices = HKMesh_getVertices(mesh);
		HKArrayVec3* normals  = HKMesh_getNormals(mesh);

		HKUnknown_release((HKUnknown*)vertices);
		HKUnknown_release((HKUnknown*)normals );
	}
	HKUnknown_release((HKUnknown*)mesh);
	return 0;
}
