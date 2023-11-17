#include "test0.h"
#include <hikari/value_array.h>
#include <hikari/object_array.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <cassert>
#include <iostream>
int main()
{
	// 通常型のOBJECT用のRefPtr
	HKRefPtr<HKMesh> mesh = HKRefPtr<hk::Mesh>::create();// 0->1->0
	{
		// 配列型のOBJECT用のRefPtr
		auto vertices = hk::TypeArrayRefPtr<hk::Vec3>::create();
		vertices.resize(3);
		vertices[0]   = HKVec3_create3(-1.0f, -1.0f, 0.0f);
		vertices[1]   = HKVec3_create3(+3.0f, -1.0f, 0.0f);
		vertices[2]   = HKVec3_create3(-1.0f, +3.0f, 0.0f);

		// 配列型のOBJECT用のRefPtr
		auto normals  = hk::ArrayRefPtr<hk::Array<hk::Vec3>>::create();
		normals.resize(3);
		normals[0]    = HKVec3_create3(0.0f, 0.0f, 1.0f);
		normals[1]    = HKVec3_create3(0.0f, 0.0f, 1.0f);
		normals[2]    = HKVec3_create3(0.0f, 0.0f, 1.0f);

		mesh->setVertices(vertices.get());
		mesh->setNormals(normals.get());
	}
	{
		// 配列型のOBJECT用のRefPtr
		auto vertices  = hk::TypeArrayRefPtr<hk::Vec3>   (mesh->getVertices() );
		auto normals   = hk::TypeArrayRefPtr<hk::Vec3>   (mesh->getNormals()  );
		auto submeshes = hk::TypeArrayRefPtr<hk::SubMesh>(mesh->getSubMeshes());
	}
	assert(mesh->getVertexCount() == 3);
	assert(mesh->getTopologoy(0)  == HKMeshTopologyTriangles);
	assert(mesh->getSubMeshCount()==1);
	assert(mesh->hasNormal());
	
	return 0;
}

