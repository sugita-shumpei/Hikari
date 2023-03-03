#ifndef HIKARI_MATH_MAT__H
#define HIKARI_MATH_MAT__H
#include <Hikari/TypeTraits.h>
#include <Hikari/Preprocessor.h>
#include <Hikari/Math/Utils.h>
namespace Hikari
{
	template <typename T> struct TMat2x2;
	template <typename T> struct TMat3x3;
	template <typename T> struct TMat4x4;

	template <typename T> struct TMat2x2 {
		HIKARI_CXX11_CONSTEXPR TMat2x2() noexcept :m_Values{ 0 } {}
		HIKARI_CXX11_CONSTEXPR TMat2x2(T v00, T v10, T v01, T v11) noexcept :m_Values{ v00,v10,v01,v11 } {}
		explicit HIKARI_CXX11_CONSTEXPR TMat2x2(T s) noexcept :m_Values{ s,T(0),T(0),s } {}

		HIKARI_CXX11_CONSTEXPR TMat2x2(const TMat2x2& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1],v.m_Values[2],v.m_Values[3] } {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TMat2x2& v) noexcept -> TVec4 & = default;

		HIKARI_CXX14_CONSTEXPR auto operator+=(const TMat2x2& v) noexcept -> TMat2x2&;
		HIKARI_CXX14_CONSTEXPR auto operator-=(const TMat2x2& v) noexcept -> TMat2x2&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(const TMat2x2& v) noexcept -> TMat2x2&;
		HIKARI_CXX14_CONSTEXPR auto operator/=(const TMat2x2& v) noexcept -> TMat2x2&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TMat2x2&;
		HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TMat2x2&;

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_CXX11_CONSTEXPR auto operator()(size_t col_idx, size_t row_idx) const noexcept -> const T& { return m_Values[2 * col_idx + row_idx]; }
		HIKARI_CXX11_CONSTEXPR auto operator()(size_t col_idx, size_t row_idx)       noexcept ->       T& { return m_Values[2 * col_idx + row_idx]; }


	private:
		T m_Values[2 * 2];
	};
	template <typename T> struct TMat3x3 {
		HIKARI_CXX11_CONSTEXPR TMat3x3() noexcept :m_Values{ 0 } {}
		HIKARI_CXX11_CONSTEXPR TMat3x3(T v00, T v10, T v20, T v01, T v11, T v21, T v02, T v12, T v22) noexcept :m_Values{ v00,v10,v20,v01,v11,v21,v02,v12,v22 } {}
		explicit HIKARI_CXX11_CONSTEXPR TMat3x3(T s) noexcept :m_Values{ s,T(0),T(0),s } {}

		HIKARI_CXX11_CONSTEXPR TMat3x3(const TMat3x3& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1],v.m_Values[2],v.m_Values[3],v.m_Values[4],v.m_Values[5],v.m_Values[6],v.m_Values[7],v.m_Values[8]} {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TMat3x3& v) noexcept -> TVec4 & = default;

		HIKARI_CXX14_CONSTEXPR auto operator+=(const TMat3x3& v) noexcept -> TMat3x3&;
		HIKARI_CXX14_CONSTEXPR auto operator-=(const TMat3x3& v) noexcept -> TMat3x3&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(const TMat3x3& v) noexcept -> TMat3x3&;
		HIKARI_CXX14_CONSTEXPR auto operator/=(const TMat3x3& v) noexcept -> TMat3x3&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TMat3x3&;
		HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TMat3x3&;

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_CXX11_CONSTEXPR auto operator()(size_t col_idx, size_t row_idx) const noexcept -> const T& { return m_Values[3 * col_idx + row_idx]; }
		HIKARI_CXX11_CONSTEXPR auto operator()(size_t col_idx, size_t row_idx)       noexcept ->       T& { return m_Values[3 * col_idx + row_idx]; }


	private:
		T m_Values[3 * 3];
	};
	template <typename T> struct TMat4x4 {

	private:
		T m_Values[4 * 4];
	};


	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator+=(const TMat2x2& v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator-=(const TMat2x2& v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator*=(const TMat2x2& v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator/=(const TMat2x2& v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator*=(T v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TMat2x2<T>::operator/=(T v) noexcept -> TMat2x2&
	{
		return HIKARI_CXX14_CONSTEXPR auto();
	}

}


#endif
