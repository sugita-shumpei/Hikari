#include <Test0.h>
int main(int argc, const char* argv[])
{
	using namespace Hikari;
	constexpr auto v1 = TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3));
	constexpr auto v2 = AnyOf(TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3)));
	constexpr auto v3 = AddOf(TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3)));
	assert(Normalize(TVec3<float>(4))== Normalize(TVec3<float>::Ones()));
	
	std::cout << TMat3<float>(
		11, 2, 1,
		2,  3, 1,
		31, 0, 2
	).Det() << std::endl;

	std::cout << TMat4<float>(
		11, 2, 1,3,
		 2, 3, 1,4,
		31, 0, 2,5,
		 6, 7, 8,9
		).Det() << std::endl;

	ShowMat4(TMat4<float>(
		11, 2, 1, 3,
		2, 3, 1, 4,
		31, 0, 2, 5,
		6, 7, 8, 9
	).Inverse());

	ShowMat4(TMat4<float>(
		11, 2, 1, 3,
		2, 3, 1, 4,
		31, 0, 2, 5,
		6, 7, 8, 9
		).Inverse()* TMat4<float>(
			11, 2, 1, 3,
			2, 3, 1, 4,
			31, 0, 2, 5,
			6, 7, 8, 9));
	return 0;
}