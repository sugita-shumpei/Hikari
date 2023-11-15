#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/quat.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/sphere.h>
#include <hikari/transform_graph.h>
#include <iostream>
int main()
{
	// 参照カウント1
	auto ref  = HKRefPtr<HKSampleObject>(new HKSampleObject());
	auto ref1 = HKRefPtr<HKSampleObject>();
	// 参照カウント2
	ref1      = ref;
	auto ref2 = std::move(ref1);
	auto ref3 = HKRefPtr<HKSampleObject>(new HKSampleObject());
	// 参照カウント3
	ref3      = ref2;
	auto ref4 = HKRefPtr<HKUnknown>();	
	// 参照カウント4
	ref3.queryInterface(ref4);
	constexpr auto v2 = HKVec2_create2(1.0f,2.0f);
	static_assert(HKVec3::ones() == HK_TYPE_INITIALIZER(HKVec3,1.0f,1.0f,1.0f),"");
	static_assert(HKVec3::unitY().cross(HKVec3::unitZ()) == HKVec3::unitX(),"");
	static_assert(HKVec3::unitZ().cross(HKVec3::unitX()) == HKVec3::unitY(),"");
	static_assert(HKVec3::unitX().cross(HKVec3::unitY()) == HKVec3::unitZ(),"");
	static_assert(HKMat2x2(1.0f, 2.0f, 3.0f, -4.0f) == HKMat2x2(1.0f, 2.0f, 3.0f, -4.0f).inverse().inverse(), "");
	static_assert(HKMat3x3(1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 6.0f, 3.0f, 8.0f, 9.0f) == HKMat3x3(1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 6.0f, 3.0f, 8.0f, 9.0f).inverse().inverse(), "");

	HKVec3 q1 = HKQuat::euler(+30.0f, -10.0f,+20.0f).eulerAngles();
	HKVec3 q2 = HKQuat::euler(-30.0f, +10.0f,-30.0f).eulerAngles();

	auto ref5 = HKRefPtr<HKTransformGraph>::create();
	{
		ref5->setLocalPosition(HKVec3(1.0f, 1.0f, 1.0f));
		ref5->setLocalScaling (HKVec3(2.0f, 2.0f, 2.0f));

		auto sph5_1 = HKRefPtr<HKSphere>::create(HKVec3(1.0),3.0f);
		auto ref5_1 = HKRefPtr<HKTransformGraphNode>::create(); 
		ref5_1->setLocalPosition(HKVec3(1.0f, 0.0f, 0.0f));
		auto sph5_2 = HKRefPtr<HKSphere>::create(HKVec3(2.0,0.0f,0.0f), 1.0f);
		auto ref5_2 = HKRefPtr<HKTransformGraphNode>::create(); 
		ref5_2->setLocalPosition(HKVec3(0.0f, 1.0f, 0.0f));
		auto ref5_3 = HKRefPtr<HKTransformGraphNode>::create(); 
		ref5_3->setLocalPosition(HKVec3(0.0f, 0.0f, 1.0f));

		ref5->setChild(ref5_1.get()); 
		ref5->setChild(ref5_2.get());
		ref5->setChild(ref5_3.get()); 

		ref5_1->setObject(sph5_1.get());
		ref5_2->setObject(sph5_2.get());
		ref5_3->setObject(sph5_1.get());

		auto v5_1 = ref5_1->getPosition(); 
		auto v5_2 = ref5_2->getPosition(); 
		auto v5_3 = ref5_3->getPosition(); 

		auto p5_1 = ref5_1->transformPoint(HKVec3::unitX());
		auto p5_2 = ref5_2->transformPoint(HKVec3::unitX());
		auto p5_3 = ref5_3->transformPoint(HKVec3::unitX());

	}

	return 0;
}

