#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
#include <hikari/math/quat.h>
#include <hikari/math/aabb.h>
#include <hikari/shape/sphere.h>
#include <hikari/transform_graph.h>
#include <stdio.h>
#include <assert.h>
int main() {
	
	{
		HKUnknown* unknown = (HKUnknown*)HKSampleObject_create();
		{
			HKSampleObject* sample_object = NULL;
			if (HKUnknown_queryInterface((HKUnknown*)unknown, HK_OBJECT_TYPEID_SampleObject, (void**)&sample_object)) {
				HKSampleObject_setName(sample_object, "sample_object");
				printf("%s\n", HKSampleObject_getName(sample_object));
				{
					HKMat2x2 m1 = HKMat2x2_create4(1.0f, 2.0f, 3.0f, -4.0f);
					HKMat2x2 m2 = HKMat2x2_inverse(&m1);
					HKMat2x2 m3 = HKMat2x2_mul(&m1, &m2);
					HKF32    d1 = HKMat2x2_determinant(&m1);
					HKF32    d2 = HKMat2x2_determinant(&m2);
					HKF32    d3 = d1 * d2;
					printf("d1=%f\n", d1);
					printf("d2=%f\n", d2);
					printf("d3=%f\n", d3);
				}
				{
					HKMat3x3 m1 = HKMat3x3_create9(1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 6.0f, 3.0f, 8.0f, 9.0f);
					HKMat3x3 m2 = HKMat3x3_inverse(&m1);
					HKMat3x3 m3 = HKMat3x3_mul(&m1, &m2);
					HKF32    d1 = HKMat3x3_determinant(&m1);
					HKF32    d2 = HKMat3x3_determinant(&m2);
					HKF32    d3 = d1 * d2;
					printf("d1=%f\n", d1);
					printf("d2=%f\n", d2);
					printf("d3=%f\n", d3);
				}
				{
					HKMat4x4 m1 = HKMat4x4_create16(1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 6.0f, 3.0f, 8.0f, 9.0f, 13.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);
					HKMat4x4 m2 = HKMat4x4_inverse(&m1);
					HKMat4x4 m3 = HKMat4x4_mul(&m1, &m2);
					HKF32    d1 = HKMat4x4_determinant(&m1);
					HKF32    d2 = HKMat4x4_determinant(&m2);
					HKF32    d3 = d1 * d2;
				}
				HKVec2 v = HKVec2_create2(1.0f, 1.0f);
				printf("v.len=%f\n", HKVec2_length(&v));
				HKVec2 v2 = HKVec2_create2(1.0f, 1.0001f);
				assert(HKVec2_equal_withEps(&v, &v2, 1.0e-4f));
				assert(HKVec2_equal_withEps(&v, &v2, 1.0e-4f));

				HKQuat qx = HKQuat_euler(90.0f, 0.0f, 0.0f);
				HKVec3 vx = HKQuat_eulerAngles(&qx);
				HKQuat qy = HKQuat_euler(0.0f, 90.0f, 0.0f);
				HKVec3 vy = HKQuat_eulerAngles(&qy);
				HKQuat qz = HKQuat_euler(0.0f, 0.0f, 90.0f);
				HKVec3 vz = HKQuat_eulerAngles(&qz);

				printf("vx=[%f, %f, %f]\n", vx.x, vx.y, vx.z);
				printf("vy=[%f, %f, %f]\n", vy.x, vy.y, vy.z);
				printf("vz=[%f, %f, %f]\n", vz.x, vz.y, vz.z);

				HKVec3 ex = HKVec3_unitX();
				HKVec3 ey = HKVec3_unitY();
				HKVec3 ez = HKVec3_unitZ();

				HKVec3 exx = HKQuat_rotateVector(&qx, &ex);
				HKVec3 exy = HKQuat_rotateVector(&qx, &ey);
				HKVec3 exz = HKQuat_rotateVector(&qx, &ez);
				printf("exx=[%f, %f, %f]\n", exx.x, exx.y, exx.z);
				printf("exy=[%f, %f, %f]\n", exy.x, exy.y, exy.z);
				printf("exz=[%f, %f, %f]\n", exz.x, exz.y, exz.z);
				HKVec3 eyx = HKQuat_rotateVector(&qy, &ex);
				HKVec3 eyy = HKQuat_rotateVector(&qy, &ey);
				HKVec3 eyz = HKQuat_rotateVector(&qy, &ez);
				printf("eyx=[%f, %f, %f]\n", eyx.x, eyx.y, eyx.z);
				printf("eyy=[%f, %f, %f]\n", eyy.x, eyy.y, eyy.z);
				printf("eyz=[%f, %f, %f]\n", eyz.x, eyz.y, eyz.z);
				HKVec3 ezx = HKQuat_rotateVector(&qz, &ex);
				HKVec3 ezy = HKQuat_rotateVector(&qz, &ey);
				HKVec3 ezz = HKQuat_rotateVector(&qz, &ez);
				printf("ezx=[%f, %f, %f]\n", ezx.x, ezx.y, ezx.z);
				printf("ezy=[%f, %f, %f]\n", ezy.x, ezy.y, ezy.z);
				printf("ezz=[%f, %f, %f]\n", ezz.x, ezz.y, ezz.z);

				HKQuat q1 = HKQuat_euler(30.0f, 46.0f, 51.0f);
				HKMat3x3 m3x3 = HKQuat_toMat3x3(&q1);
				HKQuat   q2 = HKQuat_mat3x3(&m3x3);
				printf("q1=[%f, %f, %f, %f]\n", q1.x, q1.y, q1.z, q1.w);
				printf("q2=[%f, %f, %f, %f]\n", q2.x, q2.y, q2.z, q2.w);
			}
			HKUnknown_release((HKUnknown*)sample_object);
		}
		HKUnknown_release(unknown);
	}
	{
		HKUnknown* unknown = (HKUnknown*)HKTransformGraph_create();
		{
			HKTransformGraph* graph = NULL;
			if (HKUnknown_queryInterface(unknown, HK_OBJECT_TYPEID_TransformGraph, (void**)&graph)) {

				HKTransformGraphNode* node = HKTransformGraphNode_create();
				HKTransformGraph_setChild(graph, node);

				HKUnknown_release((HKUnknown*)node);
				HKUnknown_release((HKUnknown*)graph);
			}
		}
		HKUnknown_release(unknown);
	}
	{
		HKUnknown* unknown = (HKUnknown*)HKSphere_create();
		{
			HKSphere* sphere = NULL;
			if (HKUnknown_queryInterface(unknown, HK_OBJECT_TYPEID_Sphere, (void**)&sphere)) {
				HKVec3 center1 = HKSphere_getCenter(sphere);
				HKF32  radius1 = HKSphere_getRadius(sphere);
				HKSphere_setCenter(sphere, HKVec3_create3(1.0f,1.0f,1.0f));
				HKSphere_setRadius(sphere, 2.0f);
				HKVec3 center2 = HKSphere_getCenter(sphere);
				HKF32  radius2 = HKSphere_getRadius(sphere);
				HKU32 cnt = HKUnknown_release((HKUnknown*)sphere);
			}
		}
		{

			HKU32 cnt = HKUnknown_release(unknown);
		}

	}
	return 0;
}
