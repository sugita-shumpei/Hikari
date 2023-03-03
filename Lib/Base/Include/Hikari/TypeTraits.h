#ifndef HIKARI_BASE_TYPE_TRAITS__H
#define HIKARI_BASE_TYPE_TRAITS__H
#include <Hikari/Preprocessor.h>
namespace Hikari
{
	template<bool B>
	struct BoolType {
		static constexpr bool value = B;
	};
	struct TrueType:BoolType<true> {
	};
	struct FalseType :BoolType<false> {
	};

	template<bool B, class T = void>
	struct EnableIf {};

	template<class T>
	struct EnableIf<true, T> { using type = T; };

	template<bool B, class T, class F>
	struct TypeIf {
		using type = T;
	};
	template<class T, class F>
	struct TypeIf<false,T,F> {
		using type = F;
	};

	template<class ...Args>
	struct TypeSwitch;
	
	template<bool B, typename T>
	struct TypeCase;

	template<typename T,class ...Args>
	struct TypeSwitch<TypeCase<true, T>, Args...>{ using type = T; };

	template<typename T, class ...Args>
	struct TypeSwitch<TypeCase<false,T>, Args...>: TypeSwitch<Args...>{};

	template<typename T, typename DefaultType>
	struct TypeSwitch<TypeCase<true, T>, DefaultType> { using type = T; };

	template<typename T, typename DefaultType>
	struct TypeSwitch<TypeCase<false, T>, DefaultType> { using type = DefaultType; };

	template<typename IntegerT, IntegerT... Seq>
	struct IntegerSequence {
		using value_type = IntegerT;
		static HIKARI_CXX11_CONSTEXPR std::size_t size() noexcept {
			return sizeof...(Seq);
		}
	};
	namespace Impl
	{
		template<typename IntegerT, IntegerT Beg, IntegerT End, IntegerT... Seq>
		struct MakeIntegerSequenceImpl
		 : MakeIntegerSequenceImpl<IntegerT, Beg+1, End,Seq..., Beg>{};
		template<typename IntegerT, IntegerT Beg, IntegerT... Seq>
		struct MakeIntegerSequenceImpl<IntegerT,Beg, Beg,Seq...> {
			using type = IntegerSequence<IntegerT, Seq...>;
		};
	}
	template<typename IntegerT, IntegerT N>
	using MakeIntegerSequence = typename Impl::MakeIntegerSequenceImpl<IntegerT, IntegerT(0), N>::type;

}
#endif