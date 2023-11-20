#ifndef HK_MATH_COMMON__H
#define HK_MATH_COMMON__H
#include "../platform.h"
#include "../data_type.h"

#define HK_M_E      2.71828182845904523536   // e
#define HK_M_LOG2E  1.44269504088896340736   // log2(e)
#define HK_M_LOG10E 0.434294481903251827651  // log10(e)
#define HK_M_LN2    0.693147180559945309417  // ln(2)
#define HK_M_LN10   2.30258509299404568402   // ln(10)
#define HK_M_PI     3.14159265358979323846   // pi
#define HK_M_PI_2   1.57079632679489661923     // pi/2
#define HK_M_PI_4   0.785398163397448309616    // pi/4
#define HK_M_1_PI   0.318309886183790671538    // 1/pi
#define HK_M_2_PI   0.636619772367581343076    // 2/pi

#if defined(FLT_MAX)
#define HK_FLT_MAX FLT_MAX
#else
#define HK_FLT_MAX 3.402823466e+38f
#endif

#if defined(FLT_MIN)
#define HK_FLT_MIN FLT_MIN
#else
#define HK_FLT_MIN 1.175494351e-38f
#endif

#define HK_M_DEG_2_RAD 1.745329251994329576924e-2
#define HK_M_RAD_2_DEG 57.2957795130823208768

#define HK_POW2(X) ((X)*(X))
#define HK_POW3(X) ((X)*(X)*(X))

HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_fmaxf(HKF32 v0, HKF32 v1) HK_CXX_NOEXCEPT {
	return (v0 > v1) ? v0: v1;
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_fminf(HKF32 v0, HKF32 v1) HK_CXX_NOEXCEPT {
	return (v0 < v1) ? v0 : v1;
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_fclampf(HKF32 v, HKF32 a, HKF32 b) HK_CXX_NOEXCEPT {
	return HKMath_fminf(HKMath_fmaxf(v,a),b);
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_pow2f(HKF32 x)
{
	return x * x;
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_pow3f(HKF32 x)
{
	return x * x * x;
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_pow4f(HKF32 x)
{
	return x * x * x * x;
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32 HKMath_pow5f(HKF32 x)
{
	return x * x * x * x * x;
}

#if !defined(__CUDACC__)
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_powf (HKF32 x, HKF32 y);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_sqrtf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_cbrtf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_hypotf(HKF32 x, HKF32 y);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_cosf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_sinf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_tanf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_acosf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_asinf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_atanf(HKF32 v);
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_atan2f(HKF32 y, HKF32 x);
#else
HK_INLINE HKF32 HKMath_powf (HKF32 x, HKF32 y) { return powf (x, y); }
HK_INLINE HKF32 HKMath_sqrtf(HKF32 v) { return sqrtf(v); }
HK_INLINE HKF32 HKMath_cbrtf(HKF32 v) { return cbrtf(v); }
HK_INLINE HKF32 HKMath_hypotf(HKF32 x, HKF32 y) { return hypotf(x, y); }
HK_INLINE HKF32 HKMath_cosf(HKF32 v) { return cosf(v); }
HK_INLINE HKF32 HKMath_sinf(HKF32 v) { return sinf(v); }
HK_INLINE HKF32 HKMath_tanf(HKF32 v) { return tanf(v); }
HK_INLINE HKF32 HKMath_acosf(HKF32 v) { return acosf(v); }
HK_INLINE HKF32 HKMath_asinf(HKF32 v) { return asinf(v); }
HK_INLINE HKF32 HKMath_atanf(HKF32 v) { return atanf(v); }
HK_INLINE HKF32 HKMath_atan2f(HKF32 y, HKF32 x) { return atan2f(y,x); }
#endif

#endif
