#include <iostream>
#include <array>
#include <Hikari/Math/Vec.h>
#include <Hikari/Math/Qua.h>
template<typename T,size_t N>
struct Hikari::TupleTraits<std::array<T, N>>
{
	using data_type = T;
	static HIKARI_CXX11_CONSTEXPR auto dims() noexcept ->size_t { return N; }

	template<typename ...Ts>
	static HIKARI_CXX11_CONSTEXPR auto Make(Ts... args) noexcept -> std::array<T, N> {
		return std::array<T, N>{args...};
	}

	template<size_t IDX> 
	static HIKARI_CXX11_CONSTEXPR auto Get(const std::array < T, N>& v) noexcept -> T {
		return v[IDX];
	}

	template<size_t IDX>
	static HIKARI_CXX14_CONSTEXPR auto Get(  std::array < T, N>& v) noexcept -> T& {
		return v[IDX];
	}
};
template<typename T> struct Hikari::VecTraits<std::array<T, 2>> : Hikari::TupleTraits<std::array<T, 2>>
{

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[0]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[0]; }

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[1]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[1]; }
};
template<typename T> struct Hikari::VecTraits<std::array<T, 3>> : Hikari::TupleTraits<std::array<T, 3>>
{

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[0]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[0]; }

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[1]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[1]; }

	static HIKARI_CXX11_CONSTEXPR auto Z(const std::array < T, N>& v) const noexcept -> T { return v[2]; }
	static HIKARI_CXX14_CONSTEXPR auto Z(std::array < T, N>& v)       noexcept -> T& { return v[2]; }

};
template<typename T> struct Hikari::VecTraits<std::array<T, 4>> : Hikari::TupleTraits<std::array<T, 4>>
{

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[0]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[0]; }

	static HIKARI_CXX11_CONSTEXPR auto X(const std::array < T, N>& v) const noexcept -> T { return v[1]; }
	static HIKARI_CXX14_CONSTEXPR auto X(std::array < T, N>& v)       noexcept -> T& { return v[1]; }

	static HIKARI_CXX11_CONSTEXPR auto Z(const std::array < T, N>& v) const noexcept -> T { return v[2]; }
	static HIKARI_CXX14_CONSTEXPR auto Z(std::array < T, N>& v)       noexcept -> T& { return v[2]; }

	static HIKARI_CXX11_CONSTEXPR auto W(const std::array < T, N>& v) const noexcept -> T { return v[3]; }
	static HIKARI_CXX14_CONSTEXPR auto W(std::array < T, N>& v)       noexcept -> T& { return v[3]; }
};
int main(int argc, const char** argv)
{
	constexpr auto v  = Hikari::MakeIntegerSequence<int, 3>();
	static_assert(Hikari::Impl::Tuple_Equal(
	Hikari::Impl::Tuple_Add(
		  std::array<float, 3>{1, 2, 3},
		  std::array<float, 3>{1, 2, 3}
	),    std::array<float, 3>{2, 4, 6}
	), "" );
	static_assert(Hikari::Impl::Tuple_Equal(
		Hikari::Impl::Tuple_Mul(
		std::array<float, 3>{1, 2, 3},
		3.0f
		), std::array<float, 3>{3, 6, 9}
	), "");
	static_assert(Hikari::Impl::Tuple_Equal(
		Hikari::Impl::Tuple_Div(
			std::array<float, 3>{1, 2, 3},
			3.0f
		), std::array<float, 3>{1.0f/3.0f, 2.0f/3.0f, 1.0f}
	), "");
	static_assert(Hikari::Impl::Vec_Dot(
		std::array<float, 3>{1, 2, 3},
		std::array<float, 3>{1, 2, 3}
	)==14,"");
	static_assert(Hikari::Impl::Tuple_LengthSqr(
		std::array<float, 3>{1, 2, 3}
	) == 14, "");

	static_assert(Hikari::Impl::Tuple_Equal(
		std::array<float, 3>{1, 2, 3},
		std::array<float, 3>{1, 2, 3}
	),"");
	static_assert(Hikari::Impl::Tuple_Equal(Hikari::Impl::Tuple_Equals(
		std::array<float, 3>{1, 2, 3},
		std::array<float, 3>{1, 3, 3}
	),  std::array<float, 3>{1, 0, 1}),"");
	static_assert(Hikari::Impl::Tuple_AllOf(
		std::array<float, 3>{1, 1, 1}
	),"");
	static_assert(Hikari::Impl::Tuple_AnyOf(
		Hikari::Impl::Vec_UnitX<std::array<float, 3>>()
	),"");
	static_assert(!Hikari::Impl::Tuple_NoneOf(
		std::array<float, 3>{1, 0, 0}
	),"");

	static_assert(Hikari::Impl::Tuple_Equal(
		Hikari::Impl::Tuple_Clamp(
		std::array<float, 3>{0, 2, 4},
		std::array<float, 3>{1, 1, 1},
		std::array<float, 3>{3, 3, 3}
	), std::array<float, 3>{1, 2, 3}),"");

	const auto v12 = Hikari::Impl::Vec_Normalize(
		std::array<float, 3>{0, 1, 1}
	);
	const auto v13 = Hikari::TVec2<float>(1,1).Normalized();
	std::cout << v12[0] << "-" << v12[1] << "-" << v12[2] << std::endl;
	std::cout << v13[0] << "-" << v13[1] << std::endl;

	static_assert(Hikari::TVec2<float>(1, 1).Dot(Hikari::TVec2<float>(2, 3)) == 5.0f, "");
	static_assert(Hikari::Impl::Tuple_Equal(Hikari::TQua<float>(0, 2, 2, 2).Inversed(), Hikari::TQua<float>(0, -2.0f / 12.0f, -2.0f / 12.0f, -2.0f / 12.0f)), "");
}