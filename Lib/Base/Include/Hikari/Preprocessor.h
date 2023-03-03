#ifndef HIKARI_BASE_PREPROCESSOR__H
#define HIKARI_BASE_PREPROCESSOR__H

#if __cplusplus >= 202002L
#define HIKARI_CXX20_CONSTEXPR constexpr
#else
#define HIKARI_CXX20_CONSTEXPR 
#endif

#if __cplusplus >= 201703L
#define HIKARI_CXX17_CONSTEXPR constexpr
#else
#define HIKARI_CXX17_CONSTEXPR 
#endif

#if __cplusplus >= 201402L
#define HIKARI_CXX14_CONSTEXPR constexpr
#else
#define HIKARI_CXX14_CONSTEXPR 
#endif

#if __cplusplus >= 201103L
#define HIKARI_CXX11_CONSTEXPR constexpr
#else
#define HIKARI_CXX11_CONSTEXPR 
#endif

#ifdef __CUDA_ARCH__
#define HIKARI_INLINE __forceinline__
#else
#define HIKARI_INLINE inline
#endif

#endif
