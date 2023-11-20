#include "test0.h"
#include <hikari/value_array.h>
#include <hikari/object_array.h>
#include <hikari/ref_ptr.h>
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/obj_mesh.h>
#include <cassert>
#include <iostream>

int main()
{
	HKRefPtr<HKObjMesh> objmesh = HKRefPtr<hk::ObjMesh>::create();
	if (objmesh->loadFile(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Sponza\sponza.obj)")) {

	}

	HKRefPtr<HKMesh>    mesh    = HKRefPtr<hk::Mesh>::create();
	mesh->copy(objmesh.get());


	return 0;
}
