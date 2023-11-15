#include "test0.h"
#include <hikari/math/vec.h>
#include <hikari/math/matrix.h>
int main()
{
	auto ref  = HKRefPtr<HKSampleObject>(new HKSampleObject());
	auto ref1 = HKRefPtr<HKSampleObject>();
	ref1      = ref;
	auto ref2 = std::move(ref1);
	auto ref3 = HKRefPtr<HKSampleObject>(new HKSampleObject());
	ref3      = ref2;
	auto ref4 = HKRefPtr<HKUnknown>();
	ref3.queryInterface(ref4);
	constexpr auto v2 = HKVec2_create2(1.0f,2.0f);
	constexpr auto m2 = HKMat2x2::identity();
	constexpr auto m3 = HKMat3x3::identity();
	constexpr auto m4 = HKMat4x4::identity();
	constexpr auto m5 = HKMat2x2::zeros();
	constexpr auto m6 = HKMat3x3::zeros();
	constexpr auto m7 = HKMat4x4::zeros();
	return 0;
}

