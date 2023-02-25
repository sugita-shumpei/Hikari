#ifndef HIKARI_TEST_TEST0__H
#define HIKARI_TEST_TEST0__H
#include <Hikari/Math/Vec.h>
#include <Hikari/Math/Mat.h>
#include <cmath>
#include <cassert>
#include <iostream>

template<typename T>
void ShowMat3(const Hikari::TMat3<T>& v)
{
	std::cout << v.Col0_0() << "," << v.Col0_1() << "," << v.Col0_2() << std::endl;
	std::cout << v.Col1_0() << "," << v.Col1_1() << "," << v.Col1_2() << std::endl;
	std::cout << v.Col2_0() << "," << v.Col2_1() << "," << v.Col2_2() << std::endl;
}

template<typename T>
void ShowMat4(const Hikari::TMat4<T>& v)
{
	std::cout << v.Col0_0() << "," << v.Col0_1() << "," << v.Col0_2() << "," << v.Col0_3() << std::endl;
	std::cout << v.Col1_0() << "," << v.Col1_1() << "," << v.Col1_2() << "," << v.Col1_3() << std::endl;
	std::cout << v.Col2_0() << "," << v.Col2_1() << "," << v.Col2_2() << "," << v.Col2_3() << std::endl;
	std::cout << v.Col3_0() << "," << v.Col3_1() << "," << v.Col3_2() << "," << v.Col3_3() << std::endl;
}
#endif
