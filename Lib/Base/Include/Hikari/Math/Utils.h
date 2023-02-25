#ifndef HIKARI_MATH_UTILS__H
#define HIKARI_MATH_UTILS__H
#include <Hikari/Preprocessors.h>
#include <cmath>
namespace Hikari
{
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Max(T a, T b) noexcept -> T { return a >= b ? a : b; }
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Min(T a, T b) noexcept -> T { return a <= b ? a : b; }
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Clamp(T v, T l, T h) noexcept -> T { return Min(Max(v,l),h); }

	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Pow2(T v) noexcept -> T { return v * v; }
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Pow3(T v) noexcept -> T { return v * v * v; }
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Pow4(T v) noexcept -> T { return v * v * v * v; }
	template<typename T, bool Cond = std::is_arithmetic_v<T>> HIKARI_CXX11_CONSTEXPR auto Pow5(T v) noexcept -> T { return v * v * v * v * v; }

	namespace Impl {
		template < typename T >
		constexpr T Sqrt_Detail(T s, T x, T prev)
		{
			return x != prev ?
				Sqrt_Detail(s, (x + s / x) / 2.0, x) : x;
		}
	}

	template<typename T> T Sqrt(T v)
	{
		return std::sqrt(v);
	}


}
#endif
