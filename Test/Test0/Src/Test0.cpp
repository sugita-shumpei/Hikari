#include <Test0.h>
int main(int argc, const char* argv[])
{
	using namespace Hikari;
	constexpr auto v1 = TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3));
	constexpr auto v2 = AnyOf(TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3)));
	constexpr auto v3 = AddOf(TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3)));
	assert(Normalize(TVec3<float>(4))== Normalize(TVec3<float>::Ones()));
	return 0;
}