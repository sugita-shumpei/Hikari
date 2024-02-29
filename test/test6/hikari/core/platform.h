#pragma once

#if defined(GL_core_profile)
#define HK_LANG_GLSL     1
#endif

#if defined(__HLSL_VERSION)
#define HK_LANG_HLSL     1
#endif

#if defined(__CUDACC__)
#define HK_LANG_CUDA_CXX 1
#define HK_LANG_CXX      1
#define HK_INLINE        __forceinline__
#elif defined(__cplusplus) 
#define HK_LANG_CXX      1
#define HK_LANG_CXX_HOST 1
#define HK_INLINE        inline
#else
#define HK_INLINE        
#endif

#if defined(_WIN32)
#define HK_PLATFORM_WINDOWS   1
#elif defined(__linux__)
#define HK_PLATFORM_LINUX     1
#elif defined(__APPLE__)
#define HK_PLATFORM_APPLE     1
#elif defined(__ANDROID__)
#define HK_PLATFORM_ANDROID   1
#endif

#ifndef NDEBUG
#define HK_DEBUG 1
#endif
