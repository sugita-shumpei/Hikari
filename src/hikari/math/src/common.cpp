#define HK_DLL_EXPORT 
#include <hikari/math/common.h>
#if defined(__cplusplus)
#if !defined(__CUDACC__)
#include <cmath>
#endif
#else
#include <math.h>
#endif
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_powf(HKF32 x, HKF32 y)
{
    return powf(x,y);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_sqrtf(HKF32 v)
{
    return sqrtf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_cbrtf(HKF32 v) {
    return cbrtf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_hypotf(HKF32 x, HKF32 y) {
    return hypotf(x, y);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_cosf(HKF32 v)
{
    return cosf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_sinf(HKF32 v)
{
    return sinf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_tanf(HKF32 v)
{
    return tanf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_acosf(HKF32 v)
{
    return acosf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_asinf(HKF32 v)
{
    return asinf(v);
}
HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_atanf(HKF32 v)
{
    return atanf(v);
}

HK_EXTERN_C HKF32 HK_DLL HK_API HKMath_atan2f(HKF32 y, HKF32 x) {
    return atan2f(y, x);
}