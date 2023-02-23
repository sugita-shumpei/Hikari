#ifndef HK_CORE_COMMON_MACRO_H
#define HK_CORE_COMMON_MACRO_H
#include <version>
#if __cplusplus >= 202002L
#define HK_CXX20_CONSTEXPR constexpr
#else
#define HK_CXX20_CONSTEXPR 
#endif
#if __cplusplus >= 201703L
#define HK_CXX17_CONSTEXPR constexpr
#else
#define HK_CXX17_CONSTEXPR 
#endif
#if __cplusplus >= 201402L
#define HK_CXX14_CONSTEXPR constexpr
#else
#define HK_CXX14_CONSTEXPR 
#endif
#endif
