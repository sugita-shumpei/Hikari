#ifndef HK_PLATFORM__H
#define HK_PLATFORM__H

#if !defined(__CUDACC__)
#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif
#if !defined(HK_STATIC)
#define HK_DYNAMIC
#endif
#else
#define HK_STATIC
#undef  HK_DYNAMIC
#endif

#if !defined(__CUDACC__)
#if defined(_WIN32) || defined(_WIN64)
#define HK_API      __stdcall
#define HK_PFN_PROC FARPROC
#if defined(HK_DYNAMIC) && !defined(HK_RUNTIME_LOAD)
#if defined(HK_DLL_EXPORT)
#define HK_DLL __declspec(dllexport)
#else
#define HK_DLL __declspec(dllimport)
#endif
#else
#define HK_DLL
#endif
#else
#define HK_API 
#define HK_PFN_PROC 
#define HK_DLL
#endif
#else
#define HK_API 
#define HK_PFN_PROC 
#define HK_DLL 
#endif

#if defined(HK_DYNAMIC)
#if defined(HK_RUNTIME_LOAD)
#define HK_DLL_FUNCTION            typedef
#define HK_DLL_FUNCTION_NAME(NAME) (HK_API*Pfn_##NAME)
#else
#define HK_DLL_FUNCTION            HK_DLL
#define HK_DLL_FUNCTION_NAME(NAME) HK_API NAME
#endif
#else
#define HK_DLL_FUNCTION            
#define HK_DLL_FUNCTION_NAME(NAME) HK_API NAME
#endif

// CUDA
#ifndef __CUDACC__
#define HK_INLINE inline
#else
#define HK_INLINE __forceinline__
#endif
// C++
#if defined(__cplusplus)
// C++11
#if __cplusplus >= 201103L
#define HK_CXX11_CONSTEXPR constexpr
#define HK_CXX_NOEXCEPT    noexcept
#else
#define HK_CXX11_CONSTEXPR 
#define HK_CXX_NOEXCEPT    
#endif
// C++14
#if __cplusplus >= 201402L
#define HK_CXX14_CONSTEXPR constexpr
#else
#define HK_CXX14_CONSTEXPR 
#endif
// C++17
#if __cplusplus >= 201703L
#define HK_CXX17_CONSTEXPR constexpr
#else
#define HK_CXX17_CONSTEXPR 
#endif
// C++
#define HK_CXX_CONSTEXPR HK_CXX11_CONSTEXPR
#define HK_EXTERN_C extern "C"
#define HK_TYPE_INITIALIZER(TYPE,...) HK_TYPE_INITIALIZER_IMPL(TYPE,__VA_ARGS__)
#define HK_TYPE_INITIALIZER_IMPL(TYPE,...) TYPE{__VA_ARGS__}
#define HK_REF &
#else
#define HK_CXX_CONSTEXPR 
#define HK_CXX11_CONSTEXPR 
#define HK_CXX14_CONSTEXPR 
#define HK_CXX17_CONSTEXPR 
#define HK_CXX_NOEXCEPT    
#define HK_EXTERN_C 
#define HK_TYPE_INITIALIZER(TYPE,...) (TYPE){__VA_ARGS__}
#define HK_REF 
#endif

// Static Assertion: https://qiita.com/h1day/items/053800d8bd81fb1f77ca
#define HK_STATIC_ASSERT(CND, MSG) \
    typedef char static_assertion_##MSG[(!!(CND))*2-1]
#define HK_COMPILE_TIME_ASSERT_3(X,L) \
    HK_STATIC_ASSERT(X,static_assertion_at_line##L)
#define HK_COMPILE_TIME_ASSERT_2(X,L) \
    HK_COMPILE_TIME_ASSERT_3(X,L)
#define HK_COMPILE_TIME_ASSERT(X) \
    HK_COMPILE_TIME_ASSERT_2(X,__LINE__)

#if defined(__cplusplus)
#define HK_NAMESPACE_TYPE_ALIAS(TYPE) namespace hk { typedef ::HK##TYPE TYPE; }
#else
#define HK_NAMESPACE_TYPE_ALIAS(TYPE) 
#endif

#define HK_MATH_FUNCTION_TABLE_MEMBER_DEFINITION(FUNCTION) \
Pfn##_##FUNCTION pfn_##FUNCTION

#endif