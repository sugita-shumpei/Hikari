#ifndef HIKARI_MATH_QUA__H
#define HIKARI_MATH_QUA__H
#include <Hikari/TypeTraits.h>
#include <Hikari/Preprocessor.h>
#include <Hikari/Math/Utils.h>
namespace Hikari
{
	template <typename T> struct TQua;

	template <typename T> struct TQua {
		HIKARI_CXX11_CONSTEXPR TQua() noexcept :m_Values{ 0 } {}

		HIKARI_CXX11_CONSTEXPR TQua(T x, T y, T z) noexcept :m_Values{ T(0),x,y,z} {}
		HIKARI_CXX11_CONSTEXPR TQua(T w, T x, T y, T z) noexcept :m_Values{ w,x,y,z } {}
		explicit HIKARI_CXX11_CONSTEXPR TQua(T s) noexcept: m_Values { s, T(0), T(0), T(0) }{}

		HIKARI_CXX11_CONSTEXPR TQua(const TQua& v) noexcept :m_Values{ v.m_Values[0],v.m_Values[1],v.m_Values[2],v.m_Values[3] } {}
		HIKARI_CXX14_CONSTEXPR auto operator=(const TQua& v) noexcept -> TQua & = default;

		HIKARI_CXX14_CONSTEXPR auto operator+=(const TQua& v) noexcept -> TQua&;
		HIKARI_CXX14_CONSTEXPR auto operator-=(const TQua& v) noexcept -> TQua&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(const TQua& v) noexcept -> TQua&;
		HIKARI_CXX14_CONSTEXPR auto operator*=(T v) noexcept -> TQua&;
		HIKARI_CXX14_CONSTEXPR auto operator/=(T v) noexcept -> TQua&;

		HIKARI_CXX11_CONSTEXPR auto operator[](size_t idx)const noexcept -> const T& { return m_Values[idx]; }
		HIKARI_CXX14_CONSTEXPR auto operator[](size_t idx) noexcept ->        T& { return m_Values[idx]; }

		HIKARI_CXX11_CONSTEXPR auto LengthSqr() const noexcept -> T;
		auto Length() const noexcept -> T;

		HIKARI_CXX11_CONSTEXPR auto Inversed() const noexcept -> TQua;

							   auto Normalized() const noexcept -> TQua;
							   void Normalize() noexcept;

		HIKARI_CXX11_CONSTEXPR auto Data()const noexcept -> const T* { return m_Values; }
		HIKARI_CXX14_CONSTEXPR auto Data()      noexcept ->       T* { return m_Values; }

		HIKARI_CXX11_CONSTEXPR auto W() const noexcept -> T { return m_Values[0]; }
		HIKARI_CXX14_CONSTEXPR auto W() noexcept -> T& { return m_Values[0]; }
		HIKARI_CXX11_CONSTEXPR auto X() const noexcept -> T { return m_Values[1]; }
		HIKARI_CXX14_CONSTEXPR auto X() noexcept -> T& { return m_Values[1]; }
		HIKARI_CXX11_CONSTEXPR auto Y() const noexcept -> T { return m_Values[2]; }
		HIKARI_CXX14_CONSTEXPR auto Y() noexcept -> T& { return m_Values[2]; }
		HIKARI_CXX11_CONSTEXPR auto Z() const noexcept -> T { return m_Values[3]; }
		HIKARI_CXX14_CONSTEXPR auto Z() noexcept -> T& { return m_Values[3]; }

	private:
		T m_Values[4];
	};

	template <typename T> struct TupleTraits<TQua<T>>: TrueType
	{
		using data_type = T;
		static HIKARI_CXX11_CONSTEXPR auto dims() noexcept ->size_t { return 4; }

		static HIKARI_CXX11_CONSTEXPR auto Make(T w, T x, T y, T z) noexcept -> TQua<T> { return TQua<T>(w, x, y, z); }

		template<size_t IDX> static HIKARI_CXX11_CONSTEXPR auto Get(const TQua<T>& v) noexcept -> T { return GetImpl<IDX>::Eval_1(v); }
		template<size_t IDX> static HIKARI_CXX14_CONSTEXPR auto Get(TQua<T>& v) noexcept -> T& { return GetImpl<IDX>::Eval_2(v); }
	private:
		template<size_t IDX> struct GetImpl;
		template<> struct GetImpl<0> {
			static HIKARI_CXX11_CONSTEXPR auto Eval_1(const TQua<T>& v) noexcept-> data_type { return v.W(); }
			static HIKARI_CXX14_CONSTEXPR auto Eval_2(TQua<T>& v) noexcept-> data_type& { return v.W(); }
		};
		template<> struct GetImpl<1> {
			static HIKARI_CXX11_CONSTEXPR auto Eval_1(const TQua<T>& v) noexcept-> data_type { return v.X(); }
			static HIKARI_CXX14_CONSTEXPR auto Eval_2(TQua<T>& v) noexcept-> data_type& { return v.X(); }
		};
		template<> struct GetImpl<2> {
			static HIKARI_CXX11_CONSTEXPR auto Eval_1(const TQua<T>& v) noexcept-> data_type { return v.Y(); }
			static HIKARI_CXX14_CONSTEXPR auto Eval_2(TQua<T>& v) noexcept-> data_type& { return v.Y(); }
		};
		template<> struct GetImpl<3> {
			static HIKARI_CXX11_CONSTEXPR auto Eval_1(const TQua<T>& v) noexcept-> data_type { return v.Z(); }
			static HIKARI_CXX14_CONSTEXPR auto Eval_2(TQua<T>& v) noexcept-> data_type& { return v.Z(); }
		};
	};
	template <typename T> struct   QuaTraits<TQua<T>> :TupleTraits<TQua<T>> {
		static HIKARI_CXX11_CONSTEXPR auto W(const TQua<T>& v) noexcept -> T  { return QuaTraits::Get<0>(v); }
		static HIKARI_CXX14_CONSTEXPR auto W(      TQua<T>& v) noexcept -> T& { return QuaTraits::Get<0>(v); }
		static HIKARI_CXX11_CONSTEXPR auto X(const TQua<T>& v) noexcept -> T  { return QuaTraits::Get<1>(v); }
		static HIKARI_CXX14_CONSTEXPR auto X(      TQua<T>& v) noexcept -> T& { return QuaTraits::Get<1>(v); }
		static HIKARI_CXX11_CONSTEXPR auto Y(const TQua<T>& v) noexcept -> T  { return QuaTraits::Get<2>(v); }
		static HIKARI_CXX14_CONSTEXPR auto Y(      TQua<T>& v) noexcept -> T& { return QuaTraits::Get<2>(v); }
		static HIKARI_CXX11_CONSTEXPR auto Z(const TQua<T>& v) noexcept -> T  { return QuaTraits::Get<3>(v); }
		static HIKARI_CXX14_CONSTEXPR auto Z(      TQua<T>& v) noexcept -> T& { return QuaTraits::Get<3>(v); }
	};

	template<typename T>
	HIKARI_CXX14_CONSTEXPR auto TQua<T>::operator+=(const TQua& v) noexcept -> TQua&
	{
		Impl::Tuple_AddAssign(*this, v); return *this;
	}
	template<typename T>
	HIKARI_CXX14_CONSTEXPR auto TQua<T>::operator-=(const TQua& v) noexcept -> TQua&
	{
		Impl::Tuple_SubAssign(*this, v); return *this;
	}
	template<typename T>
	inline HIKARI_CXX14_CONSTEXPR auto TQua<T>::operator*=(const TQua& v) noexcept -> TQua&
	{
		Impl::Qua_MulAssign(*this, v) return *this;
	}
	template<typename T>
	HIKARI_CXX14_CONSTEXPR auto TQua<T>::operator*=(T v) noexcept -> TQua&
	{
		Impl::Qua_MulAssign(*this, v); return *this;
	}
	template<typename T>
	HIKARI_CXX14_CONSTEXPR auto TQua<T>::operator/=(T v) noexcept -> TQua&
	{
		Impl::Qua_DivAssign(*this, v); return *this;
	}

	template<typename T>
	HIKARI_CXX11_CONSTEXPR auto TQua<T>::LengthSqr() const noexcept -> T
	{
		return Impl::Tuple_LengthSqr(*this);
	}
	template<typename T>
	auto TQua<T>::Length() const noexcept -> T
	{
		return Impl::Tuple_Length(*this);
	}
	template<typename T>
	HIKARI_CXX11_CONSTEXPR auto TQua<T>::Inversed() const noexcept -> TQua<T>
	{
		return Impl::Qua_Inverse(*this);
	}

	template <typename T> auto TQua<T>::Normalized() const noexcept -> TQua<T> { return Impl::Qua_Normalize(*this); }
	template <typename T> void TQua<T>::Normalize() noexcept { return Impl::Qua_NormalizeAssign(*this); }
}


#endif
