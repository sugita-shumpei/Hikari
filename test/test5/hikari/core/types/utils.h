#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <type_traits>
#endif
#if defined (__cplusplus)
namespace hikari
{
  inline namespace core {
#endif
#if defined (__cplusplus)
    template<Bool Cond, typename T = nullptr_t>
    using enabler_t = std::enable_if_t<Cond, T>;
#endif

#if defined (__cplusplus)
  }
}
#endif
