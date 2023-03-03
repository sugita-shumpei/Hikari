#ifndef HIKARI_MATH_VEC__H
#define HIKARI_MATH_VEC__H
#include <Hikari/TypeTraits.h>
#include <Hikari/Preprocessor.h>
#include <Hikari/Math/Utils.h>
namespace Hikari
{
	template <typename T> struct TVec2;
	template <typename T> struct TVec3;
	template <typename T> struct TVec4;

	template <typename T> struct TVec2 {
		HIKARI_CXX11_CONSTEXPR TVec2() noexcept :m_Values{ 0 } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec2(T x, T y) noexcept :m_Values{ x,y } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec2(T s) noexcept :m_Values{ s,s } {}

		HIKARI_CXX11_CONSTEXPR TVec2(const TVec2& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1]} {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TVec2& v) noexcept -> TVec2& = default;

		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator+=(const TVec2& v) noexcept -> TVec2&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator-=(const TVec2& v) noexcept -> TVec2&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator*=(const TVec2& v) noexcept -> TVec2&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(const TVec2& v) noexcept -> TVec2&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TVec2&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TVec2&;

		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Dot(const TVec2<T>& v) const noexcept -> T;
		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto LengthSqr() const noexcept -> T;
		HIKARI_INLINE auto Length() const noexcept -> T;

		HIKARI_INLINE auto Normalized() const noexcept -> TVec2<T>;
		HIKARI_INLINE void Normalize () noexcept;

		HIKARI_CXX11_CONSTEXPR auto Data()const noexcept -> const T* { return m_Values; }
		HIKARI_CXX14_CONSTEXPR auto Data()      noexcept ->       T* { return m_Values; }

		HIKARI_CXX11_CONSTEXPR auto X() const noexcept -> T { return m_Values[0]; }
		HIKARI_CXX14_CONSTEXPR auto X() noexcept -> T& { return m_Values[0]; }
		HIKARI_CXX11_CONSTEXPR auto Y() const noexcept -> T { return m_Values[1]; }
		HIKARI_CXX14_CONSTEXPR auto Y() noexcept -> T& { return m_Values[1]; }

	private: 
		T m_Values[2];
	};
	template <typename T> struct TVec3 {
		HIKARI_CXX11_CONSTEXPR TVec3() noexcept :m_Values{ 0 } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec3(T x, T y, T z) noexcept :m_Values{ x,y,z } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec3(T s) noexcept :m_Values{ s,s,s } {}

		HIKARI_CXX11_CONSTEXPR TVec3(const TVec3& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1],v.m_Values[2] } {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TVec3& v) noexcept -> TVec3 & = default;

		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator+=(const TVec3& v) noexcept -> TVec3&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator-=(const TVec3& v) noexcept -> TVec3&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator*=(const TVec3& v) noexcept -> TVec3&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(const TVec3& v) noexcept -> TVec3&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TVec3&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TVec3&;

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T&{ return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Dot(const TVec3<T>& v) const noexcept -> T;
		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto LengthSqr() const noexcept -> T;
		HIKARI_INLINE auto Length() const noexcept -> T;

		HIKARI_INLINE auto Normalized() const noexcept -> TVec3<T>;
		HIKARI_INLINE void Normalize() noexcept;

		HIKARI_CXX11_CONSTEXPR auto Data()const noexcept -> const T* { return m_Values; }
		HIKARI_CXX14_CONSTEXPR auto Data()      noexcept ->       T* { return m_Values; }

		HIKARI_CXX11_CONSTEXPR auto X() const noexcept -> T { return m_Values[0]; }
		HIKARI_CXX14_CONSTEXPR auto X() noexcept -> T& { return m_Values[0]; }
		HIKARI_CXX11_CONSTEXPR auto Y() const noexcept -> T { return m_Values[1]; }
		HIKARI_CXX14_CONSTEXPR auto Y() noexcept -> T& { return m_Values[1]; }
		HIKARI_CXX11_CONSTEXPR auto Z() const noexcept -> T { return m_Values[2]; }
		HIKARI_CXX14_CONSTEXPR auto Z() noexcept -> T& { return m_Values[2]; }

	private:
		T m_Values[3];
	};
	template <typename T> struct TVec4 {
		HIKARI_CXX11_CONSTEXPR TVec4() noexcept :m_Values{ 0 } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec4(T x, T y, T z, T w) noexcept :m_Values{ x,y,z,w } {}
		explicit HIKARI_CXX11_CONSTEXPR TVec4(T s) noexcept :m_Values{ s,s,s,s } {}

		HIKARI_CXX11_CONSTEXPR TVec4(const TVec4& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1],v.m_Values[2],v.m_Values[3] } {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TVec4& v) noexcept -> TVec4 & = default;

		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator+=(const TVec4& v) noexcept -> TVec4&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator-=(const TVec4& v) noexcept -> TVec4&;
		HIKARI_INLINE  auto operator*=(const TVec4& v) noexcept -> TVec4&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(const TVec4& v) noexcept -> TVec4&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TVec4&;
		HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TVec4&;

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Dot(const TVec4<T>& v) const noexcept -> T;
		HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto LengthSqr() const noexcept -> T;
		auto Length() const noexcept -> T;

		auto Normalized() const noexcept -> TVec4<T>;
		void Normalize() noexcept;

		HIKARI_CXX11_CONSTEXPR auto Data()const noexcept -> const T* { return m_Values; }
		HIKARI_CXX14_CONSTEXPR auto Data()      noexcept ->       T* { return m_Values; }

		HIKARI_CXX11_CONSTEXPR auto X() const noexcept -> T { return m_Values[0]; }
		HIKARI_CXX14_CONSTEXPR auto X() noexcept -> T& { return m_Values[0]; }
		HIKARI_CXX11_CONSTEXPR auto Y() const noexcept -> T { return m_Values[1]; }
		HIKARI_CXX14_CONSTEXPR auto Y() noexcept -> T& { return m_Values[1]; }
		HIKARI_CXX11_CONSTEXPR auto Z() const noexcept -> T { return m_Values[2]; }
		HIKARI_CXX14_CONSTEXPR auto Z() noexcept -> T& { return m_Values[2]; }
		HIKARI_CXX11_CONSTEXPR auto W() const noexcept -> T { return m_Values[3]; }
		HIKARI_CXX14_CONSTEXPR auto W() noexcept -> T& { return m_Values[3]; }

	private:
		T m_Values[4];
	};

	template <typename T> struct TupleTraits<TVec2<T>>: TrueType
	{
		using data_type = T;
		static HIKARI_CXX11_CONSTEXPR auto dims() noexcept ->size_t { return 2; }

		static HIKARI_CXX11_CONSTEXPR auto Make(T x, T y) noexcept -> TVec2<T> { return TVec2<T>(x, y); }

		template<size_t IDX> static HIKARI_CXX11_CONSTEXPR auto Get(const TVec2<T>& v) noexcept -> T { return v[IDX]; }
		template<size_t IDX> static HIKARI_CXX14_CONSTEXPR auto Get(      TVec2<T>& v) noexcept -> T&{ return v[IDX]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto X(const TVec2<T>& v)  noexcept -> T { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto X(TVec2<T>& v) noexcept -> T& { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Y(const TVec2<T>& v)  noexcept -> T { return v[1]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto Y(TVec2<T>& v) noexcept -> T& { return v[1]; }
	};
	template <typename T> struct TupleTraits<TVec3<T>> : TrueType
	{
		using data_type = T;
		static HIKARI_CXX11_CONSTEXPR auto dims() noexcept ->size_t { return 3; }

		static HIKARI_CXX11_CONSTEXPR auto Make(T x, T y, T z) noexcept -> TVec3<T> { return TVec3<T>(x, y,z); }

		template<size_t IDX> static HIKARI_CXX11_CONSTEXPR auto Get(const TVec3<T>& v) noexcept -> T { return v[IDX]; }
		template<size_t IDX> static HIKARI_CXX14_CONSTEXPR auto Get(TVec3<T>& v) noexcept -> T& { return v[IDX]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto X(const TVec3<T>& v)  noexcept -> T { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto X(TVec3<T>& v) noexcept -> T& { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Y(const TVec3<T>& v)  noexcept -> T { return v[1]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto Y(TVec3<T>& v) noexcept -> T& { return v[1]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Z(const TVec3<T>& v)  noexcept -> T { return v[2]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto Z(TVec3<T>& v) noexcept -> T& { return v[2]; }
	};
	template <typename T> struct TupleTraits<TVec4<T>> : TrueType
	{
		using data_type = T;
		static HIKARI_CXX11_CONSTEXPR auto dims() noexcept ->size_t { return 4; }

		static HIKARI_CXX11_CONSTEXPR auto Make(T x, T y, T z, T w) noexcept -> TVec4<T> { return TVec4<T>(x, y, z, w); }

		template<size_t IDX> static HIKARI_CXX11_CONSTEXPR auto Get(const TVec4<T>& v) noexcept -> T { return v[IDX]; }
		template<size_t IDX> static HIKARI_CXX14_CONSTEXPR auto Get(TVec4<T>& v) noexcept -> T& { return v[IDX]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto X(const TVec4<T>& v)  noexcept -> T { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto X(TVec4<T>& v) noexcept -> T& { return v[0]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Y(const TVec4<T>& v)  noexcept -> T { return v[1]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto Y(TVec4<T>& v) noexcept -> T& { return v[1]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Z(const TVec4<T>& v)  noexcept -> T { return v[2]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto Z(TVec4<T>& v) noexcept -> T& { return v[2]; }
		static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto W(const TVec4<T>& v)  noexcept -> T { return v[3]; }
		static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto W(TVec4<T>& v) noexcept -> T& { return v[3]; }
	};

	template <typename T> struct VecTraits<TVec2<T>> :TupleTraits<TVec2<T>> {
	};
	template <typename T> struct VecTraits<TVec3<T>> :TupleTraits<TVec3<T>> {
	};
	template <typename T> struct VecTraits<TVec4<T>> :TupleTraits<TVec4<T>> {
	};

	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator+=(const TVec2<T>& v) noexcept -> TVec2<T>& { Impl::Tuple_AddAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator-=(const TVec2<T>& v) noexcept -> TVec2<T>& { Impl::Tuple_SubAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator*=(const TVec2<T>& v) noexcept -> TVec2<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator/=(const TVec2<T>& v) noexcept -> TVec2<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator*=(T v) noexcept -> TVec2<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec2<T>::operator/=(T v) noexcept -> TVec2<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }

	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator+=(const TVec3<T>& v) noexcept -> TVec3<T>& { Impl::Tuple_AddAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator-=(const TVec3<T>& v) noexcept -> TVec3<T>& { Impl::Tuple_SubAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator*=(const TVec3<T>& v) noexcept -> TVec3<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator/=(const TVec3<T>& v) noexcept -> TVec3<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator*=(T v) noexcept -> TVec3<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec3<T>::operator/=(T v) noexcept -> TVec3<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }

	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator+=(const TVec4<T>& v) noexcept -> TVec4<T>& { Impl::Tuple_AddAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator-=(const TVec4<T>& v) noexcept -> TVec4<T>& { Impl::Tuple_SubAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator*=(const TVec4<T>& v) noexcept -> TVec4<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator/=(const TVec4<T>& v) noexcept -> TVec4<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator*=(T v) noexcept -> TVec4<T>& { Impl::Tuple_MulAssign(*this, v); return *this; }
	template <typename T> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR auto TVec4<T>::operator/=(T v) noexcept -> TVec4<T>& { Impl::Tuple_DivAssign(*this, v); return *this; }

	template <typename T> HIKARI_INLINE auto TVec2<T>::Normalized() const noexcept -> TVec2<T> { return Impl::Vec_Normalize(*this); }
	template <typename T> HIKARI_INLINE auto TVec3<T>::Normalized() const noexcept -> TVec3<T> { return Impl::Vec_Normalize(*this); }
	template <typename T> HIKARI_INLINE auto TVec4<T>::Normalized() const noexcept -> TVec4<T> { return Impl::Vec_Normalize(*this); }

	template <typename T> HIKARI_INLINE void TVec2<T>::Normalize() noexcept { return Impl::Vec_NormalizeAssign(*this); }
	template <typename T> HIKARI_INLINE void TVec3<T>::Normalize() noexcept { return Impl::Vec_NormalizeAssign(*this); }
	template <typename T> HIKARI_INLINE void TVec4<T>::Normalize() noexcept { return Impl::Vec_NormalizeAssign(*this); }

	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec2<T>::Dot(const TVec2<T>& v) const noexcept -> T { return Impl::Vec_Dot(*this, v); }
	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec3<T>::Dot(const TVec3<T>& v) const noexcept -> T { return Impl::Vec_Dot(*this, v); }
	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec4<T>::Dot(const TVec4<T>& v) const noexcept -> T { return Impl::Vec_Dot(*this, v); }

	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec2<T>::LengthSqr() const noexcept -> T { return Impl::Tuple_LengthSqr(*this); }
	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec3<T>::LengthSqr() const noexcept -> T { return Impl::Tuple_LengthSqr(*this); }
	template <typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto TVec4<T>::LengthSqr() const noexcept -> T { return Impl::Tuple_LengthSqr(*this); }

	template <typename T> HIKARI_INLINE auto TVec2<T>::Length() const noexcept -> T { return Impl::Tuple_Length(*this); }
	template <typename T> HIKARI_INLINE auto TVec3<T>::Length() const noexcept -> T { return Impl::Tuple_Length(*this); }
	template <typename T> HIKARI_INLINE auto TVec4<T>::Length() const noexcept -> T { return Impl::Tuple_Length(*this); }
	
}
#endif
