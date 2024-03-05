#pragma once
#include <concepts>
#include <type_traits>
namespace hikari {
  template<typename FlagBits>
  struct FlagsTraits : std::false_type {};

  template<typename FlagBits>
  concept FlagBitsLike = std::is_enum_v<FlagBits> && requires {FlagBits::eMask;
  };

  template<FlagBitsLike FlagBits>
  struct FlagsTraits<FlagBits> : std::true_type {
    using base_type = std::underlying_type_t<FlagBits>;
    static inline constexpr base_type none_mask = base_type(0);
    static inline constexpr base_type all_mask = static_cast<base_type>(FlagBits::eMask);
  };

  template<typename FlagBits>
  struct Flags {
    using Traits = FlagsTraits<FlagBits>;
    using base_type = typename Traits::base_type;
    constexpr Flags() noexcept : m_base{ Traits::none_mask } {}
    constexpr Flags(FlagBits v) noexcept : m_base{ static_cast<base_type>(v) } {}
    explicit constexpr Flags(base_type v) noexcept : m_base{ v } {}
    constexpr Flags(const Flags& lhs) noexcept = default;
    constexpr Flags(Flags&& lhs) noexcept = default;
    constexpr Flags& operator=(const Flags& lhs) noexcept = default;
    constexpr Flags& operator=(Flags&& lhs) noexcept = default;
    constexpr Flags& operator=(FlagBits v) noexcept { m_base = static_cast<base_type>(v); return *this; }
    constexpr operator bool() const noexcept { return m_base != Traits::none_mask; }
    constexpr bool operator !() const noexcept { return m_base == Traits::none_mask; }
    explicit  constexpr operator base_type() const noexcept { return m_base; }
    constexpr Flags operator~() const noexcept { return Flags((~m_base) & Traits::all_mask); }
    constexpr Flags& operator|=(const Flags& v) noexcept {
      m_base |= v.m_base;
      return *this;
    }
    constexpr Flags& operator&=(const Flags& v) noexcept {
      m_base &= v.m_base;
      return *this;
    }
    constexpr Flags& operator|=(const FlagBits& v) noexcept {
      m_base |= static_cast<base_type>(v);
      return *this;
    }
    constexpr Flags& operator&=(const FlagBits& v) noexcept {
      m_base &= static_cast<base_type>(v);
      return *this;
    }
    constexpr Flags operator|(const Flags& v)const noexcept { return Flags(m_base | v.m_base); }
    constexpr Flags operator|(const FlagBits& v)const noexcept { return Flags(m_base | static_cast<base_type>(v)); }
    constexpr Flags operator&(const Flags& v) const noexcept { return Flags(m_base & v.m_base); }
    constexpr Flags operator&(const FlagBits& v)const noexcept { return Flags(m_base & static_cast<base_type>(v)); }
    constexpr bool operator==(const Flags<FlagBits>& lhs) const noexcept {
      return m_base == lhs.m_base;
    }
    constexpr bool operator!=(const Flags<FlagBits>& lhs) const noexcept {
      return m_base != lhs.m_base;
    }
    constexpr bool operator==(const FlagBits& lhs) const noexcept {
      return m_base == static_cast<base_type>(lhs);
    }
    constexpr bool operator!=(const FlagBits& lhs) const noexcept {
      return m_base != static_cast<base_type>(lhs);
    }
  private:
    base_type m_base;
  };
  template<typename FlagBits>
  constexpr auto operator~(const FlagBits& lhs) ->std::enable_if_t<FlagsTraits<FlagBits>::value, Flags<FlagBits>> {
    return ~Flags(lhs);
  }
  template<typename FlagBits>
  constexpr bool operator==(const FlagBits& lhs, const Flags<FlagBits>& rhs) {
    return rhs == lhs;
  }
  template<typename FlagBits>
  constexpr bool operator!=(const FlagBits& lhs, const Flags<FlagBits>& rhs) {
    return rhs != lhs;
  }
  template<typename FlagBits>
  constexpr Flags<FlagBits> operator|(const FlagBits& lhs, const Flags<FlagBits>& rhs) {
    return rhs | lhs;
  }
  template<typename FlagBits>
  constexpr Flags<FlagBits> operator&(const FlagBits& lhs, const Flags<FlagBits>& rhs) {
    return rhs & lhs;
  }
  template<typename FlagBits>
  constexpr auto operator|(const FlagBits& lhs, const FlagBits& rhs) ->std::enable_if_t<FlagsTraits<FlagBits>::value, Flags<FlagBits>> {
    return Flags(rhs) | Flags(lhs);
  }
  template<typename FlagBits>
  constexpr auto operator&(const FlagBits& lhs, const FlagBits& rhs) ->std::enable_if_t<FlagsTraits<FlagBits>::value, Flags<FlagBits>> {
    return Flags(rhs) & Flags(lhs);
  }
}
