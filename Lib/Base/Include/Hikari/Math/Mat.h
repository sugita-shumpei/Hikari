#ifndef HIKARI_MATH_MAT__H
#define HIKARI_MATH_MAT__H
#include <Hikari/Math/Vec.h>
#include <Hikari/Math/MacroUtilsDef.h>
/*m_Values[DIM*I+0]...m_Values[DIM*I+DIM-1]*/
#define HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_ROW(Z, IDX, OFF) BOOST_PP_COMMA_IF(IDX) m_Values[OFF+IDX]
/*m_Values[0*DIM+I]...m_Values[(DIM-1)*DIM+I]*/
#define HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_COL(Z, IDX, OFF) BOOST_PP_COMMA_IF(IDX) m_Values[IDX OFF]
/*ROW*/
#define HIKARI_MATH_MAT_IMPL_ELM_BY_ROW_MAJOR(DIM,ROW_IDX,COL_IDX) m_Values[DIM*ROW_IDX+COL_IDX]
/*COL*/
#define HIKARI_MATH_MAT_IMPL_ELM_BY_COL_MAJOR(DIM,COL_IDX,ROW_IDX) m_Values[DIM*ROW_IDX+COL_IDX]

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW(Z, IDX, DIM) \
HIKARI_CXX11_CONSTEXPR TVec##DIM##<T>  Row##IDX()const noexcept { return TVec##DIM##<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_ROW, DIM*IDX ) ); }; \
HIKARI_CXX14_CONSTEXPR TVec##DIM##<T>& Row##IDX() noexcept { return TVec##DIM##<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_ROW, DIM*IDX ) ); }

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL(Z, IDX, DIM) \
HIKARI_CXX11_CONSTEXPR TVec##DIM##<T> Col##IDX()const noexcept { return TVec##DIM##<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_COL,*DIM+IDX ) ); }; \
HIKARI_CXX14_CONSTEXPR TVec##DIM##<T>& Col##IDX()noexcept { return TVec##DIM##<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_COL,*DIM+IDX ) ); }

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM(DIM,ROW_IDX,COL_IDX) \
HIKARI_CXX11_CONSTEXPR T Row##ROW_IDX##_##COL_IDX##() const noexcept { return HIKARI_MATH_MAT_IMPL_ELM_BY_ROW_MAJOR(DIM,ROW_IDX,COL_IDX); }; \
HIKARI_CXX14_CONSTEXPR T& Row##ROW_IDX##_##COL_IDX##() noexcept { return HIKARI_MATH_MAT_IMPL_ELM_BY_ROW_MAJOR(DIM,ROW_IDX,COL_IDX); };

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM_WITH_DIM_2(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM(2,ROW_IDX,COL_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM_WITH_DIM_3(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM(3,ROW_IDX,COL_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM_WITH_DIM_4(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM(4,ROW_IDX,COL_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELMS_FOR_EACH_COL(Z,ROW_IDX,DIM) BOOST_PP_REPEAT(DIM, BOOST_PP_CAT(HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELM_WITH_DIM_,DIM), ROW_IDX)

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM(DIM,COL_IDX,ROW_IDX) \
HIKARI_CXX11_CONSTEXPR T Col##COL_IDX##_##ROW_IDX##() const noexcept { return HIKARI_MATH_MAT_IMPL_ELM_BY_COL_MAJOR(DIM,COL_IDX,ROW_IDX); }; \
HIKARI_CXX11_CONSTEXPR T& Col##COL_IDX##_##ROW_IDX##() noexcept { return HIKARI_MATH_MAT_IMPL_ELM_BY_COL_MAJOR(DIM,COL_IDX,ROW_IDX); }

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM_WITH_DIM_2(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM(2,COL_IDX,ROW_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM_WITH_DIM_3(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM(3,COL_IDX,ROW_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM_WITH_DIM_4(Z,COL_IDX,ROW_IDX) HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM(4,COL_IDX,ROW_IDX)
#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELMS_FOR_EACH_ROW(Z,COL_IDX,DIM) BOOST_PP_REPEAT(DIM, BOOST_PP_CAT(HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELM_WITH_DIM_,DIM), COL_IDX)

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_ROWS(DIM) \
	BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW, DIM) \
	BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ACCESSOR_ROW_ELMS_FOR_EACH_COL, DIM)

#define HIKARI_MATH_MAT_IMPL_ACCESSOR_COLS(DIM) \
	BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ACCESSOR_COL, DIM) \
	BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ACCESSOR_COL_ELMS_FOR_EACH_ROW, DIM)

#define HIKARI_MATH_MAT_IMPL_ELM_ADDS_OF_OP_MULTIPLY_VEC(Z,IDX,DIM) BOOST_PP_EXPR_IF(IDX,+) lhs1.Row##IDX##() * lhs2[IDX]
#define HIKARI_MATH_MAT_IMPL_OP_MULTIPLY_VEC(DIM) \
	template<typename T> HIKARI_CXX11_CONSTEXPR auto operator*(const TMat##DIM##<T>& lhs1, const TVec##DIM##<T>& lhs2) noexcept -> TVec##DIM##<T> { \
		return BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_ADDS_OF_OP_MULTIPLY_VEC, DIM); \
	}

#define HIKARI_MATH_MAT_IMPL_ELM_ADDS_OF_OP_MULTIPLY_MAT_VEC(Z,IDX,OFF) BOOST_PP_EXPR_IF(IDX,+) lhs1.Row##IDX##() * lhs2[OFF+IDX]
#define HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_OP_MULTIPLY_MAT_ELM(Z,IDX,DIM) BOOST_PP_COMMA_IF(IDX) BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_ADDS_OF_OP_MULTIPLY_MAT_VEC, DIM*IDX)
#define HIKARI_MATH_MAT_IMPL_OP_MULTIPLY_MAT(DIM) \
	template<typename T> HIKARI_CXX11_CONSTEXPR auto operator*(const TMat##DIM##<T>& lhs1, const TMat##DIM##<T>& lhs2) noexcept -> TMat##DIM##<T> { \
		return TMat<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_OP_MULTIPLY_MAT_ELM, DIM) ); \
	}
#define HIKARI_MATH_MAT_IMPL_OP_MUL_AND_ASSIGN_MAT(DIM) \
	template<typename T> HIKARI_CXX14_CONSTEXPR auto operator*=(const TMat##DIM##<T>& lhs2) noexcept -> TMat##DIM##<T>& { \
		const auto lhs1 = *this; \
		return *this = TMat<T>( BOOST_PP_REPEAT(DIM, HIKARI_MATH_MAT_IMPL_ELM_LIST_OF_OP_MULTIPLY_MAT_ELM, DIM) ); \
	}

#define HIKARI_MATH_MAT_IMPL_METHODS(DIM) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),+=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),-=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN_S(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),*=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_ASSIGN_S(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),/=,HIKARI_CXX14_CONSTEXPR); \
	HIKARI_MATH_MAT_IMPL_OP_MUL_AND_ASSIGN_MAT(DIM); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_EQUALITY(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MAT_IMPL_ACCESSOR_ROWS(DIM); \
	HIKARI_MATH_MAT_IMPL_ACCESSOR_COLS(DIM)

#define HIKARI_MATH_MAT_IMPL_FUNCTIONS(DIM) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),+, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),-, HIKARI_CXX11_CONSTEXPR); \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_1(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),*, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_2(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),*, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_MACRO_UTILS_IMPL_OP_ARITH_S_2(BOOST_PP_CAT(TMat,DIM),BOOST_PP_MUL(DIM,DIM),/, HIKARI_CXX11_CONSTEXPR) \
	HIKARI_MATH_MAT_IMPL_OP_MULTIPLY_MAT(DIM); \
	HIKARI_MATH_MAT_IMPL_OP_MULTIPLY_VEC(DIM)

namespace Hikari
{
	//rX = (R_XX R_YX R_ZX  0)
	//rY = (R_XY R_YY R_ZY  0)
	//rZ = (R_XZ R_YZ R_ZZ  0)
	//t  = (T_X  T_Y  T_Z   1)
	//|VX*V0+VY*V4+VZ*V8 +VW*V12|
	//|VX*V1+VY*V5+VZ*V9 +VW*V13|
	//|VX*V2+VY*V6+VZ*V10+VW*V14|
	//|VX*V3+VY*V7+VZ*V11+VW*V15|
	//|A0 A4 A8  A12||B0 B4 B8  B12|
	//|A1 A5 A9  A13||B1 B5 B9  B13|
	//|A2 A6 A10 A14||B2 B6 B10 B14|
	//|A3 A7 A11 A15||B3 B7 B11 B15|
	// 
	//|C0=B0*A0+B1*A4+B2*A8 +B3*A12, C4=B4*A0+B5*A4+B6* A8+B7*A12, B8*A0+B9*A10+B11*A8+B12*A12|
	//|C1=B0*A1+B1*A5+B2*A9 +B3*A13, C5=B4*A1+B5*A5+B6* A9+B7*A13, B8*A0+B9*A10+B11*A8+B12*A12|
	//|C2=B0*A2+B1*A6+B2*A10+B3*A14, C6=B4*A2+B5*A6+B6*A10+B7*A14, B8*A0+B9*A10+B11*A8+B12*A12|
	//|C3=B0*A3+B1*A7+B2*A11+B3*A15, C7=B4*A3+B5*A7+B6*A11+B7*A15, B8*A0+B9*A10+B11*A8+B12*A12|
	//|C0, C1, C2, C3 <- B0, B1, B2, B3

	template <typename T> struct TMat2;
	template <typename T> struct TMat3;
	template <typename T> struct TMat4;

	template <typename T> struct TMat2
	{
		HIKARI_CXX11_CONSTEXPR TMat2()noexcept :m_Values{ 0 } {}
		HIKARI_CXX11_CONSTEXPR TMat2(
			T v_00, T v_10,
			T v_01, T v_11
		) noexcept : m_Values{ 
			v_00,v_10,
			v_01,v_11 
		} {}
		HIKARI_CXX11_CONSTEXPR TMat2(
			const TVec2<T>& row0,
			const TVec2<T>& row1
		) noexcept : m_Values{
			row0[0], row0[1],
			row1[0], row1[1]
		} {}

		explicit HIKARI_CXX11_CONSTEXPR TMat2(T s) noexcept : m_Values{ 
			(T)s, (T)0, 
			(T)0, (T)s 
		} {}

		HIKARI_MATH_MAT_IMPL_METHODS(2);

		HIKARI_CXX11_CONSTEXPR T     Det()const noexcept {
			return m_Values[0] * m_Values[3] - m_Values[1] * m_Values[2];
		}
		HIKARI_CXX11_CONSTEXPR TMat2 Normalized()const noexcept {
			return Internal_MulByScalar(*this, (T)1 / Det());
		}
		HIKARI_CXX11_CONSTEXPR TMat2 Inverse()const noexcept
		{
			return Internal_MulByScalar(TMat2(m_Values[3], -m_Values[2], -m_Values[1], m_Values[0]),(T)1/Det());
		}

		T m_Values[4] = { 0 };
	private:
		HIKARI_CXX11_CONSTEXPR TMat2 Internal_MulByScalar(float s)const noexcept
		{
			return TMat2(m_Values[0] * s, m_Values[1] * s, m_Values[2] * s, m_Values[3] * s);
		}
	};

	template <typename T> struct TMat3
	{
		HIKARI_CXX11_CONSTEXPR TMat3()noexcept :m_Values{ 0 } {}
		HIKARI_CXX11_CONSTEXPR TMat3(
			T v_00, T v_10, T v_20,
			T v_01, T v_11, T v_21,
			T v_02, T v_12, T v_22
		) noexcept : m_Values{ 
			v_00, v_10, v_20,
			v_01, v_11, v_21,
			v_02, v_12, v_22
		} {}
		HIKARI_CXX11_CONSTEXPR TMat3(
			const TVec3<T>& row0,
			const TVec3<T>& row1,
			const TVec3<T>& row2
		) noexcept : m_Values{
			row0[0], row0[1], row0[2],
			row1[0], row1[1], row1[2],
			row2[0], row2[1], row2[2]
		} {}
		explicit HIKARI_CXX11_CONSTEXPR TMat3(T s) noexcept 
			: m_Values{ 
				(T)s, (T)0, (T)0, 
				(T)0, (T)s, (T)0, 
				(T)0, (T)0, (T)s 
			} {}

		HIKARI_MATH_MAT_IMPL_METHODS(3);

		T m_Values[9] = { 0 };
	private:
		HIKARI_CXX11_CONSTEXPR TMat3 Internal_MulByScalar(float s)const noexcept
		{
			return TMat3(
				m_Values[0] * s, m_Values[1] * s, m_Values[2] * s,
				m_Values[3] * s, m_Values[4] * s, m_Values[5] * s,
				m_Values[6] * s, m_Values[7] * s, m_Values[8] * s
			);
		}
	};

	template <typename T> struct TMat4
	{
		HIKARI_CXX11_CONSTEXPR TMat4()noexcept :m_Values{ 0 } {}
		HIKARI_CXX11_CONSTEXPR TMat4(
			T v_00, T v_10, T v_20, T v_30,
			T v_01, T v_11, T v_21, T v_31,
			T v_02, T v_12, T v_22, T v_32,
			T v_03, T v_13, T v_23, T v_33
		) noexcept : m_Values{
			v_00, v_10, v_20, v_30,
			v_01, v_11, v_21, v_31,
			v_02, v_12, v_22, v_32,
			v_03, v_13, v_23, v_33
		} {}
		HIKARI_CXX11_CONSTEXPR TMat4(
			const TVec4<T>& row0,
			const TVec4<T>& row1,
			const TVec4<T>& row2,
			const TVec4<T>& row3
		) noexcept : m_Values{
			row0[0], row0[1], row0[2], row0[3],
			row1[0], row1[1], row1[2], row1[3],
			row2[0], row2[1], row2[2], row2[3],
			row3[0], row3[1], row3[2], row3[3]
		} {}

		explicit HIKARI_CXX11_CONSTEXPR TMat4(
			T s
		) noexcept : m_Values{
			(T)s, (T)0, (T)0, (T)0,
			(T)0, (T)s, (T)0, (T)0,
			(T)0, (T)0, (T)s, (T)0,
			(T)0, (T)0, (T)0, (T)s,
		} {}

		HIKARI_MATH_MAT_IMPL_METHODS(4);

		T m_Values[16] = { 0 };
	private:
		HIKARI_CXX11_CONSTEXPR TMat4 Internal_MulByScalar(float s)const noexcept
		{
			return TMat4(
				m_Values[0] * s, m_Values[1] * s, m_Values[2] * s, m_Values[3] * s, 
				m_Values[4] * s, m_Values[5] * s, m_Values[6] * s, m_Values[7] * s, 
				m_Values[8] * s, m_Values[9] * s, m_Values[10]* s, m_Values[11]* s,
				m_Values[12]* s, m_Values[13]* s, m_Values[14]* s, m_Values[15]* s
			);
		}
	};

	HIKARI_MATH_MAT_IMPL_FUNCTIONS(2);
	HIKARI_MATH_MAT_IMPL_FUNCTIONS(3);
	HIKARI_MATH_MAT_IMPL_FUNCTIONS(4);
}
#include <Hikari/Math/MacroUtilsUndef.h>
#endif