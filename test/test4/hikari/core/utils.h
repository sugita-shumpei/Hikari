#pragma once
#include <type_traits>
#include <algorithm>
#include <vector>
#include <hikari/core/data_type.h>
namespace hikari {
  inline namespace core {
    template<typename To, typename From, std::enable_if_t<std::is_convertible_v<From, To>, nullptr_t> = nullptr>
    Array<To>  static_array_cast(const Array<From>& from) {
      Array<To> to; to.reserve(from.size());
      std::transform(std::begin(from), std::end(from), std::back_inserter(to), [](const auto f) { return To(f); });
      return to;
    }

    template<typename To, typename From, std::enable_if_t<std::is_convertible_v<From, To>, nullptr_t> = nullptr>
    Option<To> static_option_cast(const Option<From>& from) {
      if (!from) { return std::nullopt; }
      return To(*from);
    }

    template<typename To, typename From>
    Option<To> safe_numeric_cast(From from) {
      // もし符号付整数にキャストするなら
      if      constexpr (std::is_integral_v<To> && std::is_signed_v<To>) {
        // もとが符号付整数の場合, 範囲に収まっていればよい
        if constexpr (std::is_integral_v<From> && std::is_signed_v<From>) {
          if constexpr (sizeof(To) >= sizeof(From)) { return To(from); }
          else {
            if ((std::numeric_limits<To>::min() <= from) && (from <= std::numeric_limits<To>::max())) { return To(from); }
            return {};
          }
        }
        // もとが符号無整数の場合, 範囲に収まっていればよい
        else if constexpr (std::is_integral_v<From> && std::is_unsigned_v<From>) {
          if constexpr (sizeof(To) > sizeof(From)) { return To(from); }
          else {
            if (from <= std::numeric_limits<To>::max()) { return To(from); }
            return {};
          }
        }
        // もとが浮動小数の場合, まず整数かどうか調べる
        else if constexpr (std::is_floating_point_v<From>) {
          // 整数でないなら→失敗
          if (std::floor(from) != from) { return {}; }
          // 整数であれば, 型変換が可能
          // F32: 23
          // F64: 52
          if constexpr (std::numeric_limits<From>::digits < 8 * sizeof(To)) {
            return To(from);
          }
          else {
            if (from >= std::numeric_limits<To>::max()) { return {}; }
            return To(from);
          }
        }
        else { return {}; }
      }
      // もし符号無整数にキャストするなら
      else if constexpr (std::is_integral_v<To> && std::is_unsigned_v<To>) {
        // もとが符号無整数の場合, 範囲に収まっていればよい
        if      constexpr (std::is_integral_v<From> && std::is_unsigned_v<From>) {
          if constexpr (sizeof(To) >= sizeof(From)) { return To(from); }
          else {
            if ((std::numeric_limits<To>::min() <= from) && (from <= std::numeric_limits<To>::max())) { return To(from); }
            return {};
          }
        }
        // もとが符号付整数の場合, 0以上でなければならない
        else if constexpr (std::is_integral_v<From> && std::is_signed_v<From>) {
          if constexpr (sizeof(To) > sizeof(From)) {
            if (from < 0) { return {}; }
            return To(from);
          }
          else {
            if (from >= 0 && from <= std::numeric_limits<To>::max()) { return To(from); }
            return {};
          }
        }
        // もとが浮動小数の場合, まず整数かどうか調べる
        else if constexpr (std::is_floating_point_v<From>) {
          // 整数でないなら→失敗
          if (std::floor(from) != from) { return {}; }
          // 負の数なら失敗
          if (from < static_cast<From>(0.0f)) { return {}; }
          // 整数であれば, 型変換が可能
          // F32: 23
          // F64: 52
          if constexpr (std::numeric_limits<From>::digits < 8 * sizeof(To)) {
            return To(from);
          }
          else {
            if (from >= std::numeric_limits<To>::max()) { return {}; }
            return To(from);
          }
        }
        else { return {}; }
      }
      // もし浮動小数にキャストするなら
      else if constexpr (std::is_floating_point_v<To>) {
        // もとが整数の場合, 範囲に収まっていればよい
        if constexpr (std::is_integral_v<From> && (std::is_unsigned_v<From> || std::is_signed_v<From>)) {
          if constexpr (std::numeric_limits<To>::digits >= 8 * sizeof(From)) {
            return To(from);
          }
          else {
            auto val = To(from);
            auto iva = From(val);
            if (iva == from) {
              return val;
            }
            else {
              return std::nullopt;
            }
          }
        }
        else if constexpr (std::is_floating_point_v<From>){
          if (sizeof(From) <= sizeof(To)) {
            return To(from);
          }
          else {
            if ((std::numeric_limits<To>::min() <= from) && (from <= std::numeric_limits<To>::max())) { return To(from); }
            return {};
          }
        }
      }
      return {};
    }

  }
}
