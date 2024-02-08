#pragma once
#include <tuple>
#include <variant>
#include <vector>
#include <type_traits>
namespace hikari {
  inline namespace core {
    template<typename ...Ts>
    using Tuple = std::tuple<Ts...>;

    template<size_t I, class... Types> typename       std::tuple_element<I, Tuple<Types...>>::type& get(Tuple<Types...>& t) noexcept { return std::get<I>(t); }
    template<size_t I, class... Types> typename       std::tuple_element<I, Tuple<Types...>>::type&& get(Tuple<Types...>&& t) noexcept { return std::get<I>(std::forward<Tuple<Types...>&&>(t)); }
    template<size_t I, class... Types> typename const std::tuple_element<I, Tuple<Types...>>::type& get(const Tuple<Types...>& t) noexcept { return std::get<I>(t); }

    template<class  T, class... Types> typename       T& get(Tuple<Types...>& t) noexcept { return std::get<T>(t); }
    template<class  T, class... Types> typename       T&& get(Tuple<Types...>&& t) noexcept { return std::get<T>(std::forward<Tuple<Types...>&&>(t)); }
    template<class  T, class... Types> typename const T& get(const Tuple<Types...>& t) noexcept { return std::get<T>(t); }

    template<typename     T> struct is_tuple : std::false_type {};
    template<typename ...Ts> struct is_tuple <std::tuple<Ts...>> : std::true_type {};

    template<typename Tuple1Like, typename ...Ts> struct append_tuple;

    template<template<typename...> typename Tuple1Like, typename T, typename ...Ts>
    struct append_tuple<Tuple1Like<Ts...>, T> { using type = Tuple1Like<Ts..., T>; };

    template<typename Tuple1Like, typename T1, typename T2, typename ...Ts>
    struct append_tuple<Tuple1Like, T1, T2, Ts...> : append_tuple<typename append_tuple<Tuple1Like, T1>::type, T2, Ts...> {};

    template<template<typename...> typename Tuple1Like, typename ...Ts>
    struct append_tuple<Tuple1Like<Ts...>> { using type = Tuple1Like<Ts...>; };

    template<typename Tuple1Like, typename ...Ts> using append_tuple_t = typename append_tuple<Tuple1Like, Ts...>::type;

    namespace impl {
      template<bool both_tuple_type, typename... Tuples> struct concat_tuple;
      template<typename... Tuples>
      struct concat_tuple<true, Tuples...> {
        using type = decltype(std::tuple_cat(std::declval<Tuples&&>()...));
      };
    }

    template<typename... Tuples> struct concat_tuple : impl::concat_tuple <(bool)(is_tuple<Tuples>::value + ...), Tuples...> {};
    template<typename... Tuples> using concat_tuple_t = typename concat_tuple<Tuples...>::type;

    template<typename T, typename TupleT> struct in_tuple;

    template<template<typename...>typename TupleLike, typename T, typename ...Ts>
    struct in_tuple<T, TupleLike<Ts...>> : std::bool_constant<(bool)(std::is_same_v<Ts, T>+...)> {};

    template <typename I, typename IntegerSequenceLikeT, I Val>
    struct append_integer_sequence_1;

    template <typename I, template<typename I1, I1...> typename IntegerSequenceLikeT, I Val, I... Seq>
    struct append_integer_sequence_1<I, IntegerSequenceLikeT<I, Seq...>, Val> {
      using type = IntegerSequenceLikeT<I, Seq..., Val>;
    };

    template <typename I, typename IntegerSequenceLikeT, I... Seq>
    struct append_integer_sequence;

    template <typename I, typename IntegerSequenceLikeT>
    struct append_integer_sequence<I, IntegerSequenceLikeT> {
      using type = IntegerSequenceLikeT;
    };

    template <typename I, typename IntegerSequenceLikeT, I Seq1, I... Seq>
    struct append_integer_sequence<I, IntegerSequenceLikeT, Seq1, Seq...> :
      append_integer_sequence<I, typename  append_integer_sequence_1<I, IntegerSequenceLikeT, Seq1>::type, Seq...>
    {};

    namespace impl {
      static_assert(
        std::is_same<
        append_integer_sequence<size_t, std::index_sequence<0, 1, 2, 3>, 4, 5, 6>::type,
        std::index_sequence<0, 1, 2, 3, 4, 5, 6>
        >::value, ""
        );
    }

    template <typename IntegerSequenceLikeT1, typename IntegerSequenceLikeT2>
    struct concat_integer_sequence_2;

    template <typename I, typename IntegerSequenceLikeT1, template<typename I1, I1...> typename IntegerSequenceLikeT2, I... Seq>
    struct concat_integer_sequence_2<IntegerSequenceLikeT1, IntegerSequenceLikeT2<I, Seq...>> : append_integer_sequence<I, IntegerSequenceLikeT1, Seq...> {};

    template <typename ...IntegerSequenceLikeTs>
    struct concat_integer_sequence;

    template <typename IntegerSequenceLikeT1, typename IntegerSequenceLikeT2, typename... IntegerSequenceLikeTs>
    struct concat_integer_sequence<IntegerSequenceLikeT1, IntegerSequenceLikeT2, IntegerSequenceLikeTs...> :
      concat_integer_sequence<typename concat_integer_sequence_2<IntegerSequenceLikeT1, IntegerSequenceLikeT2>::type, IntegerSequenceLikeTs...> {};

    template <typename IntegerSequenceLikeT>
    struct concat_integer_sequence<IntegerSequenceLikeT> { using type = IntegerSequenceLikeT; };


    template<typename I, I Val, typename IntegerSequenceLikeT>
    struct find_integer_sequence;
    template<typename I, I Val, I Seq, template<typename I1, I1...> typename IntegerSequenceLikeT>
    struct find_integer_sequence<I, Val, IntegerSequenceLikeT<I, Seq>> : std::integral_constant<size_t, (Val == Seq) ? 0 : 1> {};
    template<typename I, I Val, I Seq1, I... Seqs, template<typename I1, I1...> typename IntegerSequenceLikeT>
    struct find_integer_sequence<I, Val, IntegerSequenceLikeT<I, Seq1, Seqs...>> : std::integral_constant<size_t, ((Val == Seq1) ? 0 : 1 + find_integer_sequence<I, Val, IntegerSequenceLikeT<I, Seqs...>>::value) > {};

    template<size_t Val, typename IndexSequenceLikeT>
    using  find_index_sequence = find_integer_sequence<size_t, Val, IndexSequenceLikeT>;


    template<typename T, typename TupleT> struct find_tuple;
    template<template<typename...>typename TupleLike, typename T, typename ...Ts>
    struct find_tuple<T, TupleLike<Ts...>> : find_integer_sequence<bool, true, std::integer_sequence<bool, std::is_same_v<Ts, T>...>> {};

    template<typename IndexSequenceLikeT>
    struct pop_integer_sequence_front;

    template<typename I, I head, I ...tail>
    struct pop_integer_sequence_front<std::integer_sequence<I, head, tail...>> {
      using type = std::integer_sequence<I, tail...>;
    };

    template<typename I, I Val, typename IntegerSequenceLikeT>
    struct in_integer_sequence : std::integral_constant<size_t, find_integer_sequence<I, Val, IntegerSequenceLikeT>::value> {};

    template<size_t Val, typename IndexSequenceLikeT>
    struct in_index_sequence : in_integer_sequence<size_t, Val, IndexSequenceLikeT> {};

    template<typename E, typename TupleT>
    struct tuple_index;

    template<typename E, template<typename...>typename TupleLike, typename T, typename ...Ts>
    struct tuple_index<E, TupleLike<T, Ts...>> : find_index_sequence<1, std::index_sequence<std::is_same_v<T, E>, std::is_same_v<Ts, E>...>> {};

    template<typename TupleT>
    using  tuple_size = std::tuple_size<TupleT>;

    template<typename TupleT>
    struct variant_from_tuple;

    template<template<typename...>typename TupleLike, typename T, typename ...Ts>
    struct variant_from_tuple<TupleLike<T, Ts...>> { using type = std::variant<T, Ts..., std::monostate>; };

    template<template<typename, typename...> typename TransformT, typename TupleT>
    struct transform_tuple;

    template<template<typename, typename...> typename TransformT, template<typename...> typename TupleLike, typename T, typename ...Ts>
    struct transform_tuple<TransformT, TupleLike<T, Ts...>> { using type = std::tuple<TransformT<T>, TransformT<Ts>...>; };

    namespace impl {
      template<size_t len, size_t idx, typename StrData, char... chs>
      struct str_data_to_integer_sequence_impl :
        str_data_to_integer_sequence_impl<len, idx + 1, StrData, chs..., StrData::value[idx]>
      {};
      template<size_t len, typename StrData, char... chs>
      struct str_data_to_integer_sequence_impl<len, len, StrData, chs...> {
        using type = std::integer_sequence<char, chs...>;
      };
    }
    template<typename StrData>
    struct str_data_to_integer_sequence :
      impl::str_data_to_integer_sequence_impl<sizeof(StrData::value) - 1, 1, StrData, StrData::value[0]>
    {};

    namespace impl {
      struct str_data_to_integer_sequence_test {
        static constexpr char value[] = "test";
      };
      struct str_data_to_integer_sequence_test_result {
        using type = str_data_to_integer_sequence<str_data_to_integer_sequence_test>::type;
      };
      static_assert(std::is_same<str_data_to_integer_sequence_test_result::type, std::integer_sequence<char, 't', 'e', 's', 't'> >::value, "");
    }

    template<typename IntegerSequenceT>
    struct integer_sequence_to_str_data;

    template<char... chs>
    struct integer_sequence_to_str_data<std::integer_sequence<char, chs...>> {
      static inline constexpr char value[] = {
        chs...,'\0'
      };
      static inline constexpr size_t len = sizeof...(chs);
    };

    template<typename T>
    struct is_vector : std::false_type {};

    template<typename ...Ts>
    struct is_vector<std::vector<Ts...>> : std::true_type {};
  }
}
