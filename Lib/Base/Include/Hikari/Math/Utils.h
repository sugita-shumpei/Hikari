#ifndef HIKARI_MATH_UTILS__H
#define HIKARI_MATH_UTILS__H
#include <Hikari/Preprocessor.h>
#include <Hikari/TypeTraits.h>
#ifndef __CUDA_ARCH__
#include <cmath>
#endif
namespace Hikari
{
	template<typename T> struct TupleTraits : FalseType {};
	template<typename T, size_t N> struct TupleNTraits :
		TypeIf<TupleTraits<T>::value,
		typename TypeIf<TupleTraits<T>::dims() == N,
		TupleTraits<T>,
		FalseType>::type,
		FalseType
		>::type
	{};

	template<typename T> struct VecTraits : FalseType {};
	template<typename T, size_t N> struct VecNTraits :
		TypeIf<VecTraits<T>::value,
		typename TypeIf<VecTraits<T>::dims() == N,
		VecTraits<T>,
		FalseType>::type,
		FalseType
		>::type
	{};

	template<typename T> struct MatTraits : FalseType {};
	template<typename T, size_t N /*Col*/, size_t M /*Row*/> struct MatNxMTraits :
		TypeIf<MatTraits<T>::value,
		(typename TypeIf<MatTraits<T>::cols() == N) && (typename TypeIf<MatTraits<T>::rows() == N),
		MatTraits<T>,
		FalseType>::type,
		FalseType
		>::type
	{};

	template<typename T> struct QuaTraits : FalseType {};

	namespace Impl {
		template<typename T, size_t N, template <typename V, size_t IDX> typename VoidFuncDecl >   struct TupleNTraits_Apply_Void {
			using data_type = typename TupleTraits<T>::data_type;

			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple() noexcept -> T
			{
				return Impl_Apply_To_Tuple(Hikari::MakeIntegerSequence<size_t, N>());
			}
		private:
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					VoidFuncDecl<data_type, Indices>::Eval()...
				);
			}
		};
		template<typename T, size_t N, template <typename V> typename UnaryFuncDecl>   struct TupleNTraits_Apply_Unary
		{
			using data_type = typename TupleTraits<T>::data_type;
			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Apply_To_Write(T&       v) noexcept{
				return Impl_Apply_To_Write(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple(const T& v) noexcept -> T
			{
				return Impl_Apply_To_Tuple(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Sum(const T& v) noexcept -> data_type
			{
				return Impl_Apply_To_Sum(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Mul(const T& v) noexcept -> data_type
			{
				return Impl_Apply_To_Mul(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolAnd(const T& v) noexcept -> bool
			{
				return Impl_Apply_To_BoolAnd(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolOr(const T& v) noexcept -> bool
			{
				return Impl_Apply_To_BoolOr(v, Hikari::MakeIntegerSequence<size_t, N>());
			}
		private:
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Impl_Apply_To_Write(        T& v, Hikari::IntegerSequence<size_t, Indices...>) noexcept
			{
				using Swallow = int[];
				(void)Swallow {
					(UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)),0)...
				};
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(  const T& v, Hikari::IntegerSequence<size_t,Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					(UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)),0)...
				);
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Sum(    const T& v, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)) + ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(0);
				using Swallow = int[];
				(void)Swallow {
					((res = res + UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Mul(const T& v, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)) * ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(1);
				using Swallow = int[];
				(void)Swallow {
					((res = res * UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolAnd(const T& v, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)) && ...);
#elif __cplusplus >= 201402L 
				bool res = bool(true);
				using Swallow = int[];
				(void)Swallow {
					((res = res && UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolOr(const T& v, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v)) || ...);
#elif __cplusplus >= 201402L 
				bool res = bool(false);
				using Swallow = int[];
				(void)Swallow {
					((res = res || UnaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v))), (int)0)...
				};
				return res;
#endif
			}
		};
		template<typename T, size_t N, template <typename V> typename BinaryFuncDecl > struct TupleNTraits_Apply_Binary
		{
			using data_type = typename TupleTraits<T>::data_type;
			/*Tuple2Tuple*/
			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Apply_To_Write(T& v1, const T& v2) noexcept
			{
				return Impl_Apply_To_Write(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple(  const T& v1, const T& v2) noexcept -> T
			{

				return Impl_Apply_To_Tuple(v1,v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Sum(    const T& v1, const T& v2) noexcept -> data_type
			{

				return Impl_Apply_To_Sum(v1,v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Mul(    const T& v1, const T& v2) noexcept -> data_type
			{

				return Impl_Apply_To_Mul(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolAnd(const T& v1, const T& v2) noexcept -> bool
			{

				return Impl_Apply_To_BoolAnd(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolOr( const T& v1, const T& v2) noexcept -> bool
			{

				return Impl_Apply_To_BoolOr(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			/*Tuple2Scala*/
			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Apply_To_Write(T& v1, data_type v2) noexcept
			{

				return Impl_Apply_To_Write(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple(  const T& v1, data_type v2) noexcept -> T
			{

				return Impl_Apply_To_Tuple(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Sum(    const T& v1, data_type v2) noexcept -> data_type
			{

				return Impl_Apply_To_Sum(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Mul(    const T& v1, data_type v2) noexcept -> data_type
			{

				return Impl_Apply_To_Mul(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolAnd(const T& v1, data_type v2) noexcept -> bool
			{

				return Impl_Apply_To_BoolAnd(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_BoolOr( const T& v1, data_type v2) noexcept -> bool
			{

				return Impl_Apply_To_BoolOr(v1, v2, Hikari::MakeIntegerSequence<size_t, N>());
			}

		private:
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Impl_Apply_To_Write(T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept 
			{
				using Swallow = int[];
				(void)Swallow {
					(BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2)), 0)...
				};
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(const T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2))...
				);
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Sum(const T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2)) + ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(0);
				using Swallow = int[];
				(void)Swallow {
					((res = res + BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Mul(const T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2)) * ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(1);
				using Swallow = int[];
				(void)Swallow {
					((res = res * BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolAnd(const T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2)) && ...);
#elif __cplusplus >= 201402L 
				bool res = true;
				using Swallow = int[];
				(void)Swallow {
					((res = res && BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2))), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolOr(const T& v1, const T& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2)) || ...);
#elif __cplusplus >= 201402L 
				bool res = false;
				using Swallow = int[];
				(void)Swallow {
					((res = res || BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2))), (int)0)...
				};
				return res;
#endif
			}

			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Impl_Apply_To_Write(T& v1, const data_type& v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept
			{
				using Swallow = int[];
				(void)Swallow {
					(BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2), 0)...
				};
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(const T& v1, data_type v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2)...
				);
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Sum(const T& v1, data_type v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2) + ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(0);
				using Swallow = int[];
				(void)Swallow {
					((res = res + BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2)), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Mul(const T& v1, data_type v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> data_type
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2) * ...);
#elif __cplusplus >= 201402L 
				data_type res = data_type(0);
				using Swallow = int[];
				(void)Swallow {
					((res = res * BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2)), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolAnd(const T& v1, data_type v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2) && ...);
#elif __cplusplus >= 201402L 
				bool res = bool(true);
				using Swallow = int[];
				(void)Swallow {
					((res = res && BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2)), (int)0)...
				};
				return res;
#endif
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_BoolOr(const T& v1, data_type v2, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> bool
			{
#if __cplusplus >= 201703L 
				return (BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2) || ...);
#elif __cplusplus >= 201402L 
				bool res = bool(false);
				using Swallow = int[];
				(void)Swallow {
					((res = res || BinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2)), (int)0)...
				};
				return res;
#endif
			}
		};
		template<typename T, size_t N, template <typename V> typename TrinaryFuncDecl> struct TupleNTraits_Apply_Trinary
		{
			using data_type = typename TupleTraits<T>::data_type;
			/*Tuple2Tuple*/
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple(const T& v1, const T& v2, const T& v3) noexcept -> T
			{

				return Impl_Apply_To_Tuple(v1, v2, v3, Hikari::MakeIntegerSequence<size_t, N>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Tuple(const T& v1, data_type v2, data_type v3) noexcept -> T
			{

				return Impl_Apply_To_Tuple(v1, v2, v3, Hikari::MakeIntegerSequence<size_t, N>());
			}
		private:
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(const T& v1, const T& v2, const T& v3, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					TrinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), Hikari::TupleTraits<T>::Get<Indices>(v2), Hikari::TupleTraits<T>::Get<Indices>(v3))...
				);
			}

			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Tuple(const T& v1, data_type v2, data_type v3, Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::TupleTraits<T>::Make(
					TrinaryFuncDecl<data_type>::Eval(Hikari::TupleTraits<T>::Get<Indices>(v1), v2, v3)...
				);
			}
		};

		template<typename T, template <typename V, size_t IDX> typename VoidFuncDecl>    struct TupleTraits_Apply_Void :TupleNTraits_Apply_Void<T, TupleTraits<T>::dims(), VoidFuncDecl> {};
		template<typename T, template <typename V> typename UnaryFuncDecl>   struct TupleTraits_Apply_Unary  :TupleNTraits_Apply_Unary<T, TupleTraits<T>::dims(), UnaryFuncDecl> {};
		template<typename T, template <typename V> typename BinaryFuncDecl>  struct TupleTraits_Apply_Binary :TupleNTraits_Apply_Binary<T, TupleTraits<T>::dims(), BinaryFuncDecl> {};
		template<typename T, template <typename V> typename TrinaryFuncDecl> struct TupleTraits_Apply_Trinary:TupleNTraits_Apply_Trinary<T, TupleTraits<T>::dims(), TrinaryFuncDecl> {};

		template<typename T, size_t IDX> struct  OnesFuncDecl  { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(1); } };
		template<typename T, size_t IDX> struct  ZerosFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(0); } };
		template<typename T, size_t IDX> struct  UnitXFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(0); } };
		template<typename T> struct  UnitXFuncDecl<T, 0> { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(1); } };
		template<typename T, size_t IDX> struct  UnitYFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(0); } };
		template<typename T> struct  UnitYFuncDecl<T,1> { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(1); } };
		template<typename T, size_t IDX> struct  UnitZFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(0); } };
		template<typename T> struct  UnitZFuncDecl<T,2> { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(1); } };
		template<typename T, size_t IDX> struct  UnitWFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(0); } };
		template<typename T> struct  UnitWFuncDecl<T, 3> { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval() noexcept -> T { return T(1); } };

		template<typename T> struct  AddFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 + v2; } };
		template<typename T> struct  SubFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 - v2; } };
		template<typename T> struct  MulFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 * v2; } };
		template<typename T> struct  DivFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 / v2; } };

		template<typename T> struct  AddAssignFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Eval(T& v1, T v2) noexcept { return v1 += v2; } };
		template<typename T> struct  SubAssignFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Eval(T& v1, T v2) noexcept { return v1 -= v2; } };
		template<typename T> struct  MulAssignFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Eval(T& v1, T v2) noexcept { return v1 *= v2; } };
		template<typename T> struct  DivAssignFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Eval(T& v1, T v2) noexcept { return v1 /= v2; } };

		template<typename T> struct  EqualFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 == v2)?T(1):T(0); } };
		template<typename T> struct  NotEqualFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 != v2)?T(1):T(0); } };

		template<typename T> struct  LessFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 < v2)?T(1):T(0); } };
		template<typename T> struct  GreaterFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 > v2)?T(1):T(0); } };
		template<typename T> struct  LessEqualFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 <= v2)?T(1):T(0); } };
		template<typename T> struct  GreaterEqualFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1 >= v2)?T(1):T(0); } };

		template<typename T> struct  MaxFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 > v2 ? v1 : v2; } };
		template<typename T> struct  MinFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return v1 < v2 ? v1 : v2; } };

		template<typename T> struct  DistanceSqrFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2) noexcept -> T { return (v1-v2)*(v1-v2); } };

		template<typename T> struct Pow2FuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v) noexcept -> T { return v * v; } };
		template<typename T> struct Pow3FuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v) noexcept -> T { return v * v * v; } };
		template<typename T> struct Pow4FuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v) noexcept -> T { return v * v * v * v; } };
		template<typename T> struct Pow5FuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v) noexcept -> T { return v * v * v * v * v; } };
		template<typename T> struct SqrtFuncDecl { 
			static HIKARI_INLINE auto Eval(T v) noexcept -> T { 
#ifndef __CUDA_ARCH__
			using std::sqrt;
#endif
			return ::sqrt(v); } 
		};
		template<>           struct SqrtFuncDecl<float> {
			static HIKARI_INLINE auto Eval(float v) noexcept -> float {
				return ::sqrtf(v);
			}
		};
		template<typename T> struct ToBoolFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1) noexcept -> bool { return (v1 == T(1)); } };
		template<typename T> struct ClampFuncDecl { static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Eval(T v1, T v2, T v3) noexcept -> T { return MinFuncDecl<T>::Eval(MaxFuncDecl<T>::Eval(v1, v2), v3); } };


		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Clamp(const T& v1, const U& v2, const U& v3)noexcept -> T { return TupleTraits_Apply_Trinary<T,ClampFuncDecl>::Apply_To_Tuple(v1,v2,v3); }

		template<typename T>             HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Add(const T& v1, const T& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, AddFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T>             HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Sub(const T& v1, const T& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, SubFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Mul(const T& v1, const U& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, MulFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Div(const T& v1, const U& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, DivFuncDecl>::Apply_To_Tuple(v1, v2); }
		
		template<typename T>             HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Tuple_AddAssign(T& v1, const T& v2) noexcept { return TupleTraits_Apply_Binary<T, AddFuncDecl>::Apply_To_Write(v1, v2); }
		template<typename T>             HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Tuple_SubAssign(T& v1, const T& v2) noexcept { return TupleTraits_Apply_Binary<T, SubFuncDecl>::Apply_To_Write(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Tuple_MulAssign(T& v1, const U& v2) noexcept { return TupleTraits_Apply_Binary<T, MulFuncDecl>::Apply_To_Write(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void Tuple_DivAssign(T& v1, const U& v2) noexcept { return TupleTraits_Apply_Binary<T, DivFuncDecl>::Apply_To_Write(v1, v2); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Max(const T& v1, const T& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, MaxFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Min(const T& v1, const T& v2) noexcept -> T { return TupleTraits_Apply_Binary<T, MinFuncDecl>::Apply_To_Tuple(v1, v2); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_DistanceSqr(const T& v1, const T& v2) noexcept ->  typename TupleTraits<T>::data_type { return TupleTraits_Apply_Binary<T, DistanceSqrFuncDecl>::Apply_To_Sum(v1, v2); }
		template<typename T> HIKARI_INLINE                        auto Tuple_Distance(const T& v1, const T& v2) noexcept ->  typename TupleTraits<T>::data_type { return SqrtFuncDecl< typename TupleTraits<T>::data_type >::Eval(Tuple_DistanceSqr(v1,v2)); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Equal(const T& v1, const T& v2) noexcept-> bool { return TupleTraits_Apply_Binary<T, EqualFuncDecl>::Apply_To_BoolAnd(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_NotEqual(const T& v1, const T& v2) noexcept-> bool { return TupleTraits_Apply_Binary<T, NotEqualFuncDecl>::Apply_To_BoolOr(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Equals(const T& v1, const T& v2) noexcept-> T { return TupleTraits_Apply_Binary<T, EqualFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_NotEquals(const T& v1, const T& v2) noexcept-> T { return TupleTraits_Apply_Binary<T, NotEqualFuncDecl>::Apply_To_Tuple(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_AllOf(const T& v1)noexcept -> bool { return TupleTraits_Apply_Unary<T, ToBoolFuncDecl>::Apply_To_BoolAnd(v1); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_AnyOf(const T& v1)noexcept -> bool { return TupleTraits_Apply_Unary<T, ToBoolFuncDecl>::Apply_To_BoolOr(v1); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_NoneOf(const T& v1)noexcept -> bool { return !Tuple_AnyOf(v1); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Pow2(const T& v1) noexcept -> T { return TupleTraits_Apply_Unary<T, Pow2FuncDecl>::Apply_To_Tuple(v1); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Pow3(const T& v1) noexcept -> T { return TupleTraits_Apply_Unary<T, Pow3FuncDecl>::Apply_To_Tuple(v1); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Pow4(const T& v1) noexcept -> T { return TupleTraits_Apply_Unary<T, Pow4FuncDecl>::Apply_To_Tuple(v1); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_Pow5(const T& v1) noexcept -> T { return TupleTraits_Apply_Unary<T, Pow5FuncDecl>::Apply_To_Tuple(v1); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Tuple_LengthSqr(const T& v1) noexcept -> typename TupleTraits<T>::data_type { return TupleTraits_Apply_Unary<T, Pow2FuncDecl>::Apply_To_Sum(v1); }
		template<typename T>                        auto Tuple_Length(const T& v1) noexcept ->  typename TupleTraits<T>::data_type { return SqrtFuncDecl< typename TupleTraits<T>::data_type >::Eval(Tuple_LengthSqr(v1)); }

		template<typename T, size_t N/*Col*/, size_t M/*Row*/, template <typename V, size_t, size_t> typename VoidFuncDecl> struct MatNxMTraits_Apply_Void {
			using data_type = typename TupleTraits<T>::data_type;
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Mat_RowMajor() noexcept -> T
			{
				return Impl_Apply_To_Mat_RowMajor(Hikari::MakeIntegerSequence<size_t, N*M>());
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Apply_To_Mat_ColMajor() noexcept -> T
			{
				return Impl_Apply_To_Mat_ColMajor(Hikari::MakeIntegerSequence<size_t, N* M>());
			}
		private:
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Mat_RowMajor(Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::MatTraits<T>::Make_RowMajor(
					VoidFuncDecl<data_type, Indices/M, Indices%M>::Eval()...
				);
			}
			template<size_t... Indices> static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Impl_Apply_To_Mat_ColMajor(Hikari::IntegerSequence<size_t, Indices...>) noexcept -> T
			{
				return Hikari::MatTraits<T>::Make_ColMajor(
					VoidFuncDecl<data_type, Indices%N, Indices/N>::Eval()...
				);
			}
		};
		template<typename T> struct QuaTraits_FunctionDeclTable {
			using data_type = typename QuaTraits<T>::data_type;
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Mul(const T& v1, const T&  v2) noexcept -> T {
				return QuaTraits<T>::Make(
					QuaTraits<T>::W(v1) * QuaTraits<T>::W(v2) - QuaTraits<T>::X(v1) * QuaTraits<T>::X(v2) - QuaTraits<T>::Y(v1) * QuaTraits<T>::Y(v2) - QuaTraits<T>::Z(v1) * QuaTraits<T>::Z(v2),
					QuaTraits<T>::Y(v1) * QuaTraits<T>::Z(v2) - QuaTraits<T>::Z(v1) * QuaTraits<T>::Y(v2),
					QuaTraits<T>::Z(v1) * QuaTraits<T>::X(v2) - QuaTraits<T>::X(v1) * QuaTraits<T>::Z(v2),
					QuaTraits<T>::X(v1) * QuaTraits<T>::Y(v2) - QuaTraits<T>::Y(v1) * QuaTraits<T>::X(v2)
				);
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Mul(const T& v1, data_type v2) noexcept -> T {
				return Tuple_Mul(v1, v2);
			}
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Div(const T& v1, data_type v2) noexcept -> T {
				return Tuple_Div(v1, v2);
			}

			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void MulAssign(T& v1, const T&  v2) noexcept {
				T w = QuaTraits<T>::W(v1) * QuaTraits<T>::W(v2) - QuaTraits<T>::X(v1) * QuaTraits<T>::X(v2) - QuaTraits<T>::Y(v1) * QuaTraits<T>::Y(v2) - QuaTraits<T>::Z(v1) * QuaTraits<T>::Z(v2);
				T x = QuaTraits<T>::Y(v1) * QuaTraits<T>::Z(v2) - QuaTraits<T>::Z(v1) * QuaTraits<T>::Y(v2);
				T y = QuaTraits<T>::Z(v1) * QuaTraits<T>::X(v2) - QuaTraits<T>::X(v1) * QuaTraits<T>::Z(v2);
				T z = QuaTraits<T>::X(v1) * QuaTraits<T>::Y(v2) - QuaTraits<T>::Y(v1) * QuaTraits<T>::X(v2);

				QuaTraits<T>::W(v1) = w;
				QuaTraits<T>::X(v1) = x;
				QuaTraits<T>::Y(v1) = y;
				QuaTraits<T>::Z(v1) = z;
			}
			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void MulAssign(T& v1, data_type v2) noexcept {
				return Tuple_MulAssign(v1, v2);
			}
			static HIKARI_INLINE HIKARI_CXX14_CONSTEXPR void DivAssign(T& v1, data_type v2) noexcept {
				return Tuple_DivAssign(v1, v2);
			}

			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Inverse(const T& v1) noexcept -> T {
				return Make_And_Normalized(
					QuaTraits<T>::W(v1), -QuaTraits<T>::X(v1), -QuaTraits<T>::Y(v1), -QuaTraits<T>::Z(v1), Tuple_LengthSqr(v1)
				);
			}
		private:
			static HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Make_And_Normalized(data_type w, data_type x, data_type y, data_type z, data_type s) noexcept -> T {
				return QuaTraits<T>::Make(w / s, x / s, y / s, z / s);
			}
		};

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_Identity() noexcept -> T { return TupleTraits_Apply_Void<T,  OnesFuncDecl>::Apply_To_Tuple(); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_Ones () noexcept    -> T { return TupleTraits_Apply_Void<T,  OnesFuncDecl>::Apply_To_Tuple(); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_Zeros() noexcept    -> T { return TupleTraits_Apply_Void<T, ZerosFuncDecl>::Apply_To_Tuple(); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_UnitX() noexcept    -> T { return TupleTraits_Apply_Void<T, UnitXFuncDecl>::Apply_To_Tuple(); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_UnitY() noexcept    -> T { return TupleTraits_Apply_Void<T, UnitYFuncDecl>::Apply_To_Tuple(); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_UnitZ() noexcept    -> T { return TupleTraits_Apply_Void<T, UnitZFuncDecl>::Apply_To_Tuple(); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_UnitW() noexcept    -> T { return TupleTraits_Apply_Void<T, UnitWFuncDecl>::Apply_To_Tuple(); }

		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_Dot(const T& v1, const T& v2) noexcept -> typename TupleTraits<T>::data_type { return TupleTraits_Apply_Binary<T, MulFuncDecl>::Apply_To_Sum(v1, v2); }
		template<typename T> HIKARI_INLINE                        auto Vec_Normalize (const T& v) noexcept -> T { return Tuple_Mul(v,static_cast<typename TupleTraits<T>::data_type>(1)/Tuple_Length(v)); }
		template<typename T> HIKARI_INLINE                        void Vec_NormalizeAssign (T&       v) noexcept { Tuple_MulAssign(v, static_cast<typename TupleTraits<T>::data_type>(1) / Tuple_Length(v));}
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Vec_Cross(const T& v1, const T& v2)noexcept -> T {
			TupleTraits<T>::Make(
				VecNTraits<T, 3>::Y(v1) * VecNTraits<T, 3>::Z(v2) - VecNTraits<T, 3>::Z(v1) * VecNTraits<T, 3>::Y(v2),
				VecNTraits<T, 3>::Z(v1) * VecNTraits<T, 3>::X(v2) - VecNTraits<T, 3>::X(v1) * VecNTraits<T, 3>::Z(v2),
				VecNTraits<T, 3>::X(v1) * VecNTraits<T, 3>::Y(v2) - VecNTraits<T, 3>::Y(v1) * VecNTraits<T, 3>::X(v2)
			);
		}

		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Qua_Mul(const T& v1, const U& v2) noexcept -> T { return QuaTraits_FunctionDeclTable<T>::Mul(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Qua_Div(const T& v1, const U& v2) noexcept -> T { return QuaTraits_FunctionDeclTable<T>::Div(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Qua_MulAssign(const T& v1, const U& v2) noexcept{ return QuaTraits_FunctionDeclTable<T>::MulAssign(v1, v2); }
		template<typename T, typename U> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR void Qua_DivAssign(const T& v1, const U& v2) noexcept{ return QuaTraits_FunctionDeclTable<T>::DivAssign(v1, v2); }
		template<typename T> HIKARI_INLINE HIKARI_CXX11_CONSTEXPR auto Qua_Inverse(const T& v) noexcept -> T { return QuaTraits_FunctionDeclTable<T>::Inverse(v); }
		template<typename T> HIKARI_INLINE                        auto Qua_Normalize(const T& v) noexcept -> T { return Tuple_Mul(v, static_cast<typename TupleTraits<T>::data_type>(1) / Tuple_Length(v)); }
		template<typename T> HIKARI_INLINE                        void Qua_NormalizeAssign(T& v) noexcept { Tuple_MulAssign(v, static_cast<typename TupleTraits<T>::data_type>(1) / Tuple_Length(v)); }
		
	}
}
#endif