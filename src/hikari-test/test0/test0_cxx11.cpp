#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/quat.h>
#include <hikari/math/matrix.h>
#include <hikari/shape/sphere.h>
#include <hikari/transform_graph.h>
#include <iostream>
int main()
{
	// �Q�ƃJ�E���g1
	auto ref  = HKRefPtr<HKSampleObject>(HKSampleObject_create());
	auto ref1 = HKRefPtr<HKSampleObject>();
	// �Q�ƃJ�E���g2
	ref1      = ref;
	auto ref2 = HKRefPtr<HKSampleObject>(std::move(ref));

	return 0;
}

