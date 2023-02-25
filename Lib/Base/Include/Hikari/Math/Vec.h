#ifndef HIKARI_MATH_VEC__H
#define HIKARI_MATH_VEC__H
#include <Hikari/Math/Utils.h>
#include <Hikari/Math/MacroUtilsDef.h>
#include <cmath>

#define HIKARI_MATH_VEC_IMPL_ACCESSOR(NAME,IDX) \
	HIKARI_CXX11_CONSTEXPR const T& NAME()const noexcept{ return m_Values[IDX];} \
	HIKARI_CXX14_CONSTEXPR T& NAME() noexcept{ return m_Values[IDX];} 

#define HIKARI_MATH_VEC_IMPL_M_DOT(DIM) \
	HIKARI_CXX11_CONSTEXPR T Dot(const TVec##DIM& lhs) const noexcept {\
		return BOOST_PP_REPEAT(DIM,HIKARI_MATH_MACRO_UTILS_IMPL_ELM_SUM_OF_BINARY_OP,*); \
	}

#define HIKARI_MATH_VEC_IMPL_ELM_ANDS_OF_COMPARE(Z, IDX, OP) BOOST_PP_EXPR_IF(IDX,&&) (m_Values[IDX] OP lhs.m_Values[IDX])

#define HIKARI_MATH_VEC_IMPL_ELM_LIST_OF_COMPARE(Z, IDX, OP) BOOST_PP_COMMA_IF(IDX)   (m_Values[IDX] OP lhs.m_Values[IDX])?static_cast<T>(1):static_cast<T>(0)

#define HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, NAME, OP) \
	HIKARI_CXX11_CONSTEXPR TVec##DIM NAME(const TVec##DIM& lhs) const noexcept { \
		return TVec##DIM( BOOST_PP_REPEAT(DIM, HIKARI_MATH_VEC_IMPL_ELM_LIST_OF_COMPARE, OP) ); \
	}

#define HIKARI_MATH_VEC_IMPL_ARITH_OF_ELM(Z, IDX, OP) BOOST_PP_EXPR_IF(IDX,OP) lhs[IDX]

#define HIKARI_MATH_VEC_IMPL_ARITH_OF(DIM, NAME, OP) template<typename T> HIKARI_CXX11_CONSTEXPR T NAME(const TVec##DIM##<T>& lhs) noexcept { \
		return BOOST_PP_REPEAT(DIM, HIKARI_MATH_VEC_IMPL_ARITH_OF_ELM, OP); \
	}

#define HIKARI_MATH_VEC_IMPL_ALL_OF(DIM) \
	template<typename T> HIKARI_CXX11_CONSTEXPR T  AllOf(const TVec##DIM##<T>& lhs)noexcept { return (MulOf(lhs.Equal(TVec##DIM##<T>::Ones())) > T(0))?static_cast<T>(1):static_cast<T>(0); }

#define HIKARI_MATH_VEC_IMPL_ANY_OF(DIM) \
	template<typename T> HIKARI_CXX11_CONSTEXPR T  AnyOf(const TVec##DIM##<T>& lhs)noexcept { return (AddOf(lhs.Equal(TVec##DIM##<T>::Ones())) > T(0))?static_cast<T>(1):static_cast<T>(0); }

#define HIKARI_MATH_VEC_IMPL_NONE_OF(DIM) \
	template<typename T> HIKARI_CXX11_CONSTEXPR T NoneOf(const TVec##DIM##<T>& lhs)noexcept { return (AddOf(lhs.Equal(TVec##DIM##<T>::Zeros()))==T(0))?static_cast<T>(1):static_cast<T>(0); }

#define HIKARI_MATH_VEC_IMPL_LENGTH_FUNCS(DIM) \
	template<typename T>  HIKARI_CXX11_CONSTEXPR T LengthSqr(const TVec##DIM##<T>& v)noexcept { return v.Dot(v); }; \
	template<typename T>  HIKARI_CXX11_CONSTEXPR T Dot(const TVec##DIM##<T>& lhs1, const TVec##DIM##<T>& lhs2) noexcept { return lhs1.Dot(lhs2); }; \
	template<typename T>  HIKARI_CXX11_CONSTEXPR T DistanceSqr(const TVec##DIM##<T>& lhs1, const TVec##DIM##<T>& lhs2) noexcept { return LengthSqr(lhs1 - lhs2); } \
	template<typename T>  T Length(const TVec##DIM##<T>& v)noexcept { return std::sqrt(LengthSqr(v)); }; \
	template<typename T>  T Distance(const TVec##DIM##<T>& lhs1, const TVec##DIM##<T>& lhs2) noexcept { return std::sqrt(DistanceSqr(lhs1,lhs2)); }; \
	template<typename T>  TVec##DIM##<T> Normalize(const TVec##DIM##<T>& v) noexcept { return v/Length(v); }

#define HIKARI_MATH_VEC_IMPL_METHODS(DIM) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TVec,DIM),DIM,+=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TVec,DIM),DIM,-=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TVec,DIM),DIM,*=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TVec,DIM),DIM,/=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN_S(BOOST_PP_CAT(TVec,DIM),DIM,*=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN_S(BOOST_PP_CAT(TVec,DIM),DIM,/=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_EQUALITY(BOOST_PP_CAT(TVec,DIM),DIM,HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, Equal, ==); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, NotEqual, !=); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, GreaterThan,>); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, LessThan,<); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, GreaterEqual,>=); \
	HIKARI_MATH_VEC_IMPL_M_COMPARE(DIM, LessEqual, <=) \
	HIKARI_MATH_VEC_IMPL_M_DOT(DIM); \

#define HIKARI_MATH_VEC_IMPL_FUNCS(DIM) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TVec,DIM),DIM,+, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TVec,DIM),DIM,-, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TVec,DIM),DIM,*, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TVec,DIM),DIM,/, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_1(BOOST_PP_CAT(TVec,DIM),DIM,*, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_2(BOOST_PP_CAT(TVec,DIM),DIM,*, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_2(BOOST_PP_CAT(TVec,DIM),DIM,/, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_VEC_IMPL_ARITH_OF(DIM, AddOf, +); \
	HIKARI_MATH_VEC_IMPL_ARITH_OF(DIM, MulOf, *); \
	HIKARI_MATH_VEC_IMPL_ALL_OF(DIM); \
	HIKARI_MATH_VEC_IMPL_ANY_OF(DIM); \
	HIKARI_MATH_MACRO_UTILS_IMPL_BINARY_FUNC(BOOST_PP_CAT(TVec,DIM),DIM, Max, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_BINARY_FUNC(BOOST_PP_CAT(TVec,DIM),DIM, Min, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_UNARY_FUNC(BOOST_PP_CAT(TVec,DIM), DIM, Pow2, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_UNARY_FUNC(BOOST_PP_CAT(TVec,DIM), DIM, Pow3, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_UNARY_FUNC(BOOST_PP_CAT(TVec,DIM), DIM, Pow4, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_UNARY_FUNC(BOOST_PP_CAT(TVec,DIM), DIM, Pow5, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_TERNARY_FUNC(BOOST_PP_CAT(TVec,DIM), DIM, Clamp, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_TERNARY_FUNC_S2_S3(BOOST_PP_CAT(TVec,DIM), DIM, Clamp, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_VEC_IMPL_LENGTH_FUNCS(DIM)

namespace Hikari
{
	template<typename T> struct TVec2 ;
	template<typename T> struct TVec3 ;
	template<typename T> struct TVec4 ;

	template<typename T> struct TVec2
	{
		HIKARI_CXX11_CONSTEXPR TVec2()noexcept :m_Values{ 0 } {}

		HIKARI_CXX11_CONSTEXPR TVec2(T v0, T v1)noexcept :m_Values{ v0,v1 } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec2(T s)noexcept :m_Values{ s,s } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec2(const TVec3<T>& v)noexcept;
		explicit HIKARI_CXX11_CONSTEXPR TVec2(const TVec4<T>& v)noexcept;

		HIKARI_CXX11_CONSTEXPR TVec2(const TVec2&) = default;
		HIKARI_CXX11_CONSTEXPR TVec2(     TVec2&&) = default;
		HIKARI_CXX11_CONSTEXPR TVec2&operator=(const TVec2&) = default;
		HIKARI_CXX11_CONSTEXPR TVec2& operator=(TVec2&&)     = default;

		HIKARI_CXX11_CONSTEXPR TVec2 operator+()const noexcept { return *this; }
		HIKARI_CXX11_CONSTEXPR TVec2 operator-()const noexcept { return TVec2(-m_Values[0],-m_Values[1]); }

		HIKARI_MATH_VEC_IMPL_METHODS(2);

		static HIKARI_CXX11_CONSTEXPR TVec2 Zeros() {
			return TVec2{T(0),T(0)};
		}
		static HIKARI_CXX11_CONSTEXPR TVec2 Ones () {
			return TVec2{T(1),T(1)};
		}
		static HIKARI_CXX11_CONSTEXPR TVec2 UnitX() {
			return TVec2{ T(1),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec2 UnitY() {
			return TVec2{ T(0),T(1) };
		}

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx)      noexcept ->       T& { return m_Values[idx]; }

		HIKARI_MATH_VEC_IMPL_ACCESSOR(X, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(Y, 1);

		HIKARI_MATH_VEC_IMPL_ACCESSOR(R, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(G, 1);

		T m_Values[2] = {0};
	};

	template<typename T> struct TVec3
	{
		HIKARI_CXX11_CONSTEXPR TVec3()noexcept :m_Values{ 0 } {}

		HIKARI_CXX11_CONSTEXPR TVec3(T v0, T v1, T v2)noexcept :m_Values{ v0,v1,v2 } {}
		HIKARI_CXX11_CONSTEXPR TVec3(T v0, T v1)noexcept :m_Values{ v0,v1,T(0) } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec3(T s)noexcept :m_Values{ s,s,s} {}
		HIKARI_CXX11_CONSTEXPR TVec3(const TVec2<T>& v0, T v2)noexcept :m_Values{ v0[0],v0[1],v2  } {}
		HIKARI_CXX11_CONSTEXPR TVec3(T v0, const TVec2<T>& v1)noexcept :m_Values{ v0, v1[0],v1[1] } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec3(const TVec4<T>& v)noexcept;

		HIKARI_CXX11_CONSTEXPR TVec3(const TVec3&) = default;
		HIKARI_CXX11_CONSTEXPR TVec3(TVec3&&) = default;
		HIKARI_CXX11_CONSTEXPR TVec3& operator=(const TVec3&) = default;
		HIKARI_CXX11_CONSTEXPR TVec3& operator=(TVec3&&) = default;

		HIKARI_CXX11_CONSTEXPR TVec3 operator+()const noexcept { return *this; }
		HIKARI_CXX11_CONSTEXPR TVec3 operator-()const noexcept { return TVec3(-m_Values[0], -m_Values[1], -m_Values[2]); }

		HIKARI_MATH_VEC_IMPL_METHODS(3);

		HIKARI_CXX11_CONSTEXPR TVec3 Cross(const TVec3& lhs)const noexcept {
			return TVec3(
				m_Values[1] * lhs.m_Values[2] - m_Values[2] * lhs.m_Values[1],
				m_Values[2] * lhs.m_Values[0] - m_Values[0] * lhs.m_Values[2],
				m_Values[0] * lhs.m_Values[1] - m_Values[1] * lhs.m_Values[0]
			);
		}

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx)      noexcept ->       T& { return m_Values[idx]; }

		static HIKARI_CXX11_CONSTEXPR TVec3 Zeros() {
			return TVec3{ T(0),T(0),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec3 Ones () {
			return TVec3{ T(1),T(1),T(1) };
		}

		static HIKARI_CXX11_CONSTEXPR TVec3 UnitX() {
			return TVec3{ T(1),T(0),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec3 UnitY() {
			return TVec3{ T(0),T(1),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec3 UnitZ() {
			return TVec3{ T(0),T(0),T(1) };
		}
		
		HIKARI_MATH_VEC_IMPL_ACCESSOR(X, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(Y, 1);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(Z, 2);

		HIKARI_MATH_VEC_IMPL_ACCESSOR(R, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(G, 1);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(B, 2);

		HIKARI_MATH_VEC_IMPL_ACCESSOR(U, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(V, 1);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(W, 2);

		T m_Values[3] = {0};
	};

	template<typename T> struct TVec4
	{
		HIKARI_CXX11_CONSTEXPR TVec4()noexcept :m_Values{ 0 } {}

		HIKARI_CXX11_CONSTEXPR TVec4(T v0,T v1,T v2, T v3)noexcept :m_Values{ v0,v1,v2,v3 } {}
		HIKARI_CXX11_CONSTEXPR TVec4(T v0, T v1, T v2)noexcept :m_Values{ v0,v1,v2,T(0) } {}
		HIKARI_CXX11_CONSTEXPR TVec4(T v0, T v1)noexcept :m_Values{ v0,v1,T(0),T(0) } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec4(T s)noexcept :m_Values{ s,s,s,s } {}
		HIKARI_CXX11_CONSTEXPR TVec4(const TVec2<T>& v0, T v2, T v3)noexcept :m_Values{ v0[0],v0[1],v2,v3 } {}
		HIKARI_CXX11_CONSTEXPR TVec4(T v0, const TVec2<T>& v1, T v3)noexcept :m_Values{ v0,v1[0],v1[1],v3 } {}
		HIKARI_CXX11_CONSTEXPR TVec4(T v0, T v1, const TVec2<T>& v2)noexcept :m_Values{ v0,v1,v2[0],v2[1] } {}
		HIKARI_CXX11_CONSTEXPR TVec4(const TVec3<T>& v0, T v3)noexcept :m_Values{ v0[0],v0[1],v0[2],v3 } {}
		HIKARI_CXX11_CONSTEXPR TVec4(T v0, const TVec3<T>& v1)noexcept :m_Values{ v0,v1[0],v1[1],v1[2] } {}

		HIKARI_CXX11_CONSTEXPR TVec4(const TVec4&) = default;
		HIKARI_CXX11_CONSTEXPR TVec4(TVec4&&) = default;
		HIKARI_CXX11_CONSTEXPR TVec4& operator=(const TVec4&) = default;
		HIKARI_CXX11_CONSTEXPR TVec4& operator=(TVec4&&) = default;

		HIKARI_CXX11_CONSTEXPR TVec4 operator+()const noexcept { return *this; }
		HIKARI_CXX11_CONSTEXPR TVec4 operator-()const noexcept { return TVec4(-m_Values[0], -m_Values[1], -m_Values[2], -m_Values[3]); }

		HIKARI_MATH_VEC_IMPL_METHODS(4);

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx)      noexcept ->       T& { return m_Values[idx]; }

		static HIKARI_CXX11_CONSTEXPR TVec4 Zeros() {
			return TVec4{ T(0),T(0),T(0),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec4 Ones () {
			return TVec4{ T(1),T(1),T(1),T(1) };
		}

		static HIKARI_CXX11_CONSTEXPR TVec4 UnitX() {
			return TVec4{ T(1),T(0),T(0),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec4 UnitY() {
			return TVec4{ T(0),T(1),T(0),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec4 UnitZ() {
			return TVec4{ T(0),T(0),T(1),T(0) };
		}
		static HIKARI_CXX11_CONSTEXPR TVec4 UnitW() {
			return TVec4{ T(0),T(0),T(0),T(1) };
		}

		HIKARI_MATH_VEC_IMPL_ACCESSOR(X, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(Y, 1);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(Z, 2);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(W, 3);

		HIKARI_MATH_VEC_IMPL_ACCESSOR(R, 0);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(G, 1);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(B, 2);
		HIKARI_MATH_VEC_IMPL_ACCESSOR(A, 3);

		T m_Values[4] = {0};
	};

	template<typename T>  HIKARI_CXX11_CONSTEXPR TVec2<T>::TVec2(const TVec3<T>& v)noexcept :m_Values{ v[0],v[1] } {}
	template<typename T>  HIKARI_CXX11_CONSTEXPR TVec2<T>::TVec2(const TVec4<T>& v)noexcept :m_Values{ v[0],v[1] } {}
	template<typename T>  HIKARI_CXX11_CONSTEXPR TVec3<T>::TVec3(const TVec4<T>& v)noexcept :m_Values{ v[0],v[1],v[2]} {}

	HIKARI_MATH_VEC_IMPL_FUNCS(2);
	HIKARI_MATH_VEC_IMPL_FUNCS(3);
	HIKARI_MATH_VEC_IMPL_FUNCS(4);

	template<typename T>  HIKARI_CXX11_CONSTEXPR T MaxOf(const TVec2<T>& v) noexcept { return Max(v[0], v[1]); }
	template<typename T>  HIKARI_CXX11_CONSTEXPR T MaxOf(const TVec3<T>& v) noexcept { return Max(v[0], Max(v[1], v[2])); }
	template<typename T>  HIKARI_CXX11_CONSTEXPR T MaxOf(const TVec4<T>& v) noexcept { return Max(v[0], Max(v[1], Max(v[2],v[3]))); }

	template<typename T>  HIKARI_CXX11_CONSTEXPR T MinOf(const TVec2<T>& v) noexcept { return Min(v[0], v[1]); }
	template<typename T>  HIKARI_CXX11_CONSTEXPR T MinOf(const TVec3<T>& v) noexcept { return Min(v[0], Min(v[1], v[2])); }
	template<typename T>  HIKARI_CXX11_CONSTEXPR T MinOf(const TVec4<T>& v) noexcept { return Min(v[0], Min(v[1], Min(v[2], v[3]))); }

	template<typename T>  HIKARI_CXX11_CONSTEXPR TVec3<T> Cross(const TVec3<T>& lhs1, const TVec3<T>& lhs2) noexcept { return lhs1.Cross(lhs2); }

	static_assert(TVec4<float>()[0] == 0);
	static_assert(TVec4<float>::Zeros().X() == 0);
	static_assert((TVec4<int>::Ones()*2).X() == 2);
	static_assert((2*TVec4<int>::Ones()).X() == 2);

	static_assert(AnyOf(TVec4<int>(4, 4, 4).GreaterThan(TVec4<int>(4, 4, 3)))==1);
	static_assert(Max(TVec4<int>(3, 4, 5), TVec4<int>(4, 4, 3)) == TVec4<int>(4, 4, 5));
	static_assert(Min(TVec4<int>(3, 4, 5), TVec4<int>(4, 4, 3)) == TVec4<int>(3, 4, 3));
	static_assert(TVec3<int>::UnitX().Cross(TVec3<int>::UnitY())== TVec3<int>::UnitZ());
	static_assert(MaxOf(TVec3<int>(4, 3, 9)) == 9);
	static_assert(MinOf(TVec3<int>(4, 3, 9)) == 3);
	static_assert(LengthSqr(TVec3<int>(0, 3, 4)) == 25);
	static_assert(Pow4(TVec3<int>(2,2,2)) == TVec3<int>(16,16,16));
}

#undef	HIKARI_MATH_VEC_IMPL_ACCESSOR
#undef	HIKARI_MATH_VEC_IMPL_M_DOT
#undef	HIKARI_MATH_VEC_IMPL_ELM_ANDS_OF_COMPARE
#undef	HIKARI_MATH_VEC_IMPL_ELM_LIST_OF_COMPARE
#undef	HIKARI_MATH_VEC_IMPL_M_COMPARE
#undef	HIKARI_MATH_VEC_IMPL_ARITH_OF_ELM
#undef	HIKARI_MATH_VEC_IMPL_ARITH_OF
#undef	HIKARI_MATH_VEC_IMPL_ALL_OF
#undef	HIKARI_MATH_VEC_IMPL_ANY_OF
#undef	HIKARI_MATH_VEC_IMPL_NONE_OF
#undef	HIKARI_MATH_VEC_IMPL_LENGTH_FUNCS
#undef	HIKARI_MATH_VEC_IMPL_METHODS
#undef	HIKARI_MATH_VEC_IMPL_FUNCS

#include <Hikari/Math/MacroUtilsUndef.h>
#endif