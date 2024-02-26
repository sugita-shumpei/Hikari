#pragma once
#include <hikari/core/data_types.h>
#if defined(HK_LANG_CXX) 
namespace hikari {
#endif
  // FLOAT
  HK_INLINE U32  convertF32ToU32(F32 value) {
#if defined(HK_LANG_GLSL)
    return floatBitsToUint(value);
#endif
#if defined(HK_LANG_HLSL)
    return asuint(value);
#endif
#if defined(HK_LANG_CUDA_CXX)
    return __float_as_uint(value);
#endif
#if defined(HK_LANG_CXX_HOST)
    U32 v;
    memcpy(&v, &value, sizeof(v));
    return v;
#endif

  }
  HK_INLINE F32  convertU32ToF32(U32 value) {
#if defined(HK_LANG_GLSL)
    return uintBitsToFloat(value);
#endif
#if defined(HK_LANG_HLSL)
    return asfloat (value);
#endif
#if defined(HK_LANG_CUDA_CXX)
    return __uint_as_float(value);
#endif
#if defined(HK_LANG_CXX_HOST)
    F32 v;
    memcpy(&v, &value, sizeof(v));
    return v;
#endif
  }
  // HALF
  // conversion: F16   to F32
  HK_INLINE F32  convertF16X1ToF32X1(U32  value) {
#if defined(HK_LANG_GLSL)
    vec2 v = unpackHalf2x16(value);
    return v.x;
#endif
#if defined(HK_LANG_HLSL)
    return f16tof32(value & U32(0x0000FFFF));
#endif
#if defined(HK_LANG_CUDA_CXX)
    return __ushort_as_half(static_cast<U16>(value& static_cast<U32>(0x0000FFFF)));
#endif

#if defined(HK_LANG_CXX_HOST)
    return F16(F16::FromBitsTag::FromBits, static_cast<U16>(value & static_cast<U32>(0x0000FFFF)));
#endif
  }
  // conversion: F32   to F16
  HK_INLINE U32  convertF32X1ToF16X1(F32  value) {
#if defined(HK_LANG_GLSL)
    U32 v = packHalf2x16(Vec2(value,0.0f));
    return v&U32(0x0000FFFF);
#endif
#if defined(HK_LANG_HLSL)
    U32 v = f32tof16(value);
    return v & U32(0x0000FFFF);
#endif
#if defined(HK_LANG_CUDA_CXX)
    __half v = __half(value);
    return U32(__half_as_ushort(v));
#endif
#if defined(HK_LANG_CXX_HOST)
    auto v = Imath::half(value);
    return static_cast<U32>(v.bits());
#endif
  }
  // conversion: F16X2 to F32X2
  HK_INLINE Vec2 convertF16X2ToF32X2(U32  value) {
#if defined(HK_LANG_GLSL)
    return unpackHalf2x16(value);
#endif
#if defined(HK_LANG_HLSL)
    U32 upper = U32((value & U32(0xFFFF0000))>>U32(16));
    U32 lower = U32((value & U32(0x0000FFFF)));
    return Vec2(f16tof32(lower), f16tof32(upper));
#endif
#if defined(HK_LANG_CUDA_CXX)
    float upper = __half2float(__ushort_as_half(U16((value & U32(0xFFFF0000)) >> U32(16))));
    float lower = __half2float(__ushort_as_half(U16((value & U32(0x0000FFFF)))));
    return make_float2(lower, upper);
#endif

#if defined(HK_LANG_CXX_HOST)
    F32 upper = F16(F16::FromBitsTag::FromBits, U16((value & U32(0xFFFF0000)) >> U32(16)));
    F32 lower = F16(F16::FromBitsTag::FromBits, U16((value & U32(0x0000FFFF))));
    return Vec2(lower, upper);
#endif
  }
  // conversion: F32X2 to F16X2
  HK_INLINE U32  convertF32X2ToF16X2(Vec2 value) {
#if defined(HK_LANG_GLSL)
    return packHalf2x16(value);
#endif
#if defined(HK_LANG_HLSL)
    U32 lower = f32tof16(value.x);
    U32 upper = f32tof16(value.y)<<U32(16);
    return lower | upper;
#endif
#if defined(HK_LANG_CUDA_CXX)
    U32 lower = U32(__half_as_ushort(__half(value.x)));
    U32 upper = U32(__half_as_ushort(__half(value.y))) << U32(16);
    return lower | upper;
#endif
#if defined(HK_LANG_CXX_HOST)
    auto lower = U32(Imath::half(value.x).bits());
    auto upper = U32(Imath::half(value.y).bits()) << U32(16);
    return lower | upper;
#endif
  }
  //
  // UINT
  HK_INLINE I32 convertU32ToI32WithComp1(U32 value) {
    auto v = (value & U32(0x1));
    return v;
  }
  HK_INLINE U32 convertI32ToU32WithComp1(I32 value) {
    if (value >= 1) { return 1; }
    else            { return 0; }
  }

#define HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(COMP)                    \
  HK_INLINE I32 convertU32ToI32WithComp##COMP(U32 value){               \
    U32  v = (value /*& U32((U32(1)<<U32(COMP))-U32(1))*/);             \
    if (v <= U32((U32(1)<<U32(COMP-1))-U32(1))){ return v;}             \
    else { return -I32(((~v)&(U32((U32(1)<<U32(COMP-1))-U32(1))))+1); } \
  }

#define HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(COMP)                                \
  HK_INLINE U32 convertI32ToU32WithComp##COMP(I32 value){                           \
    if (value >= 0){ return U32(value) /*&U32((U32(1)<<U32(COMP-1))-U32(1))*/; }    \
    else { return (~(U32(-value)-U32(1)))&(U32((U32(1)<<U32(COMP))-U32(1))); }      \
  }

  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(2);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(2);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(3);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(3);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(4);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(4);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(5);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(5);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(6);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(6);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(7);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(7);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(8);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(8);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(9);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(9);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(10);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(10);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(11);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(11);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(12);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(12);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(13);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(13);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(14);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(14);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(15);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(15);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(16);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(16);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(17);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(17);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(18);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(18);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(19);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(19);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(20);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(20);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(21);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(21);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(22);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(22);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(23);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(23);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(24);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(24);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(25);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(25);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(26);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(26);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(27);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(27);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(28);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(28);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(29);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(29);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(30);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(30);
  HK_DEFINE_CONVERT_U32_TO_I32_WITH_COMP(31);
  HK_DEFINE_CONVERT_I32_TO_U32_WITH_COMP(31);

  HK_INLINE I32   convertU32ToI32WithComp32(U32 value){
    return I32(value);
  }
  HK_INLINE U32   convertI32ToU32WithComp32(I32 value) {
    return U32(value);
  }
  HK_INLINE I32   convertU32ToI8 (U32 value) { return convertU32ToI32WithComp8 (value); }
  HK_INLINE U32   convertI8ToU32 (I32 value) { return convertI32ToU32WithComp8 (value); }
  HK_INLINE I32   convertU32ToI16(U32 value) { return convertU32ToI32WithComp16(value); }
  HK_INLINE U32   convertI16ToU32(I32 value) { return convertI32ToU32WithComp16(value); }
  HK_INLINE I32   convertU32ToI32(U32 value) { return convertU32ToI32WithComp32(value); }
  HK_INLINE U32   convertI32ToU32(I32 value) { return convertI32ToU32WithComp32(value); }
  HK_INLINE UVec2 convertU32ToU16X2(U32   value) {
    U32 l = U32(value & U32(0x0000FFFF));
    U32 u = U32(value & U32(0xFFFF0000)) >> U32(16);
#if defined(HK_LANG_CUDA_CXX)
    return make_uint2(l, u)
#else
    return UVec2(l, u);
#endif
  }
  HK_INLINE U32   convertU16X2ToU32(UVec2 value) {
    U32 l = U32(value.x);
    U32 u = U32(value.y) >> U32(16);
    return l | u;
  }
  HK_INLINE UVec2 convertU16ToU8X2(U32    value) {
    U32 i = value;
    //U32 i = U32(value & U32(0xFFFF));
    U32 l = U32(value & i);
    U32 u = U32(value & i) >> U32(8);
#if defined(HK_LANG_CUDA_CXX)
    return make_uint2(l, u)
#else
    return UVec2(l, u);
#endif
  }
  HK_INLINE U32   convertU8X4ToU32(UVec4  value) {
    U32 l = U32(value.x/* & U32(0x00FF) */);
    U32 u = U32(value.y/* & U32(0x00FF) */) >> U32(8);
    return l | u;
  }
  HK_INLINE UVec4 convertU16ToU8X4(U32    value) {
    U32 l0 = U32(value & U32(0x000000FF));
    U32 l1 = U32(value & U32(0x0000FF00)) >> U32(8);
    U32 l2 = U32(value & U32(0x00FF0000)) >> U32(16);
    U32 l3 = U32(value & U32(0xFF000000)) >> U32(24);
#if defined(HK_LANG_CUDA_CXX)
    return make_uint4(l0, l1, l2, l3);
#else
    return UVec4(l0, l1, l2, l3);
#endif
  }
  // FP 11 AND FP 10
  HK_INLINE F32   convertUnsignedF11ToF32(U32 value) {
    // 符号(0)|指数部(5)|仮数部(6) -> 符号(1)|指数部(5)|仮数部(10)
    U32 frac     = (value & U32(0x03F));         // (6 BIT) -> (10BIT)
    U32 expo     = (value & U32(0x7C0))>>U32(6) ;// (5 BIT) 0 [1,30] 31 -15
    U32 frac_f32 = frac << U32(4);
    U32 expo_f32 = expo << U32(10);
    return convertF16X1ToF32X1(frac_f32|expo_f32);
  }
  HK_INLINE F32   convertUnsignedF10ToF32(U32 value) {
    // 符号(0)|指数部(5)|仮数部(5) -> 符号(1)|指数部(5)|仮数部(10)
    U32 frac = (value & U32(0x01F));// (5 BIT) -> (10BIT)
    U32 expo = (value & U32(0x3E0)) >> U32(5);// (5 BIT) 0 [1,30] 31 -15
    U32 frac_f32 = frac << U32(5);
    U32 expo_f32 = expo << U32(10);
    return convertF16X1ToF32X1(frac_f32 | expo_f32);
  }
  HK_INLINE U32   convertF32ToUnsignedF11(F32 value) {
    // 符号(1)|指数部(5)|仮数部(10)->符号(0)|指数部(5)|仮数部(6)
    U32 value_f16 = convertF32X1ToF16X1(value);
    U32 sign_f16  = (value_f16 & U32(0x0080))>>U32(16);
    U32 expo_f16  = (value_f16 & U32(0x7C00))>>U32(10);
    U32 frac_f16  = (value_f16 & U32(0x03FF));
    if (sign_f16 != 0) { return 0; }
    U32 frac_f11  = frac_f16>>U32(4);
    U32 expo_f11  = expo_f16<<U32(6);
    return frac_f11 | expo_f11;
  }
  HK_INLINE U32   convertF32ToUnsignedF10(F32 value) {
    // 符号(1)|指数部(5)|仮数部(10)->符号(0)|指数部(5)|仮数部(5)
    U32 value_f16 = convertF32X1ToF16X1(value);
    U32 sign_f16  = (value_f16 & U32(0x0080)) >> U32(16);
    U32 expo_f16  = (value_f16 & U32(0x7C00)) >> U32(10);
    U32 frac_f16  = (value_f16 & U32(0x03FF));
    if (sign_f16 != 0) { return 0; }
    U32 frac_f10  = frac_f16 >> U32(5);
    U32 expo_f10  = expo_f16 << U32(5);
    return frac_f10 | expo_f10;
  }
  HK_INLINE Vec3  convertR11G11B10ToVec3(U32  value) {
    //
    U32 ru = (value & U32(0x000007FF));           // 11BIT 
    U32 gu = (value & U32(0x003FF800)) >> U32(11);// 11BIT
    U32 bu = (value & U32(0xFFC00000)) >> U32(22);// 10BIT
#if defined(HK_LANG_CXX_HOST)
    static_assert((U32(0x000007FF) | U32(0x003FF800) | U32(0xFFC00000)) == U32(0xFFFFFFFF), "");
    static_assert((U32(0x003FF800) >> U32(11)) == 0x000007FF, "");
    static_assert((U32(0xFFC00000) >> U32(22)) == 0x000003FF, "");
#endif
    //
    F32 rf = convertUnsignedF11ToF32(ru);
    F32 gf = convertUnsignedF11ToF32(gu);
    F32 bf = convertUnsignedF10ToF32(bu);
    // 
#if defined(HK_LANG_CUDA_CXX)
    return make_float3(rf, gf, bf);
#else
    return Vec3(rf,gf,bf);
#endif
  }
  HK_INLINE U32   convertVec3ToR11G11B10(Vec3 value) {
    U32 r11 = convertF32ToUnsignedF11(value.x);
    U32 g11 = convertF32ToUnsignedF11(value.y);
    U32 b10 = convertF32ToUnsignedF10(value.z);
    return (b10 << U32(22)) | (g11 << U32(11)) | (r11);
  }
  // SHARED EXP
  HK_INLINE U32   convertVec3ToR9G9B9E5 (Vec3 value){
    U32 ru = convertF32ToU32(value.x);
    U32 gu = convertF32ToU32(value.y);
    U32 bu = convertF32ToU32(value.z);
    U32 sign_ru_32 = (ru&U32(0x80000000))>>U32(31);
    U32 expo_ru_32 = (ru&U32(0x7F800000))>>U32(23);
    U32 frac_ru_32 = (ru&U32(0x007FFFFF));
    U32 sign_gu_32 = (gu&U32(0x80000000))>>U32(31);
    U32 expo_gu_32 = (gu&U32(0x7F800000))>>U32(23);
    U32 frac_gu_32 = (gu&U32(0x007FFFFF));
    U32 sign_bu_32 = (bu&U32(0x80000000))>>U32(31);
    U32 expo_bu_32 = (bu&U32(0x7F800000))>>U32(23);
    U32 frac_bu_32 = (bu&U32(0x007FFFFF));
    //R: 1.ABC.....W(24) * 2^(E1-127)
    // = 0.1ABCDEFGH (9) * 2^x=[-14,15]->最大のBITを共有ビットに設定する
    //R: 0.ABCDEFGHI (9) * 2^(E1- 15)->
    U32 frag_ru_9u = ((frac_ru_32 >> U32(15)) | U32(0x100));// RGBEの指数部(9)
    U32 frag_gu_9u = ((frac_gu_32 >> U32(15)) | U32(0x100));// RGBEの指数部(9)
    U32 frag_bu_9u = ((frac_bu_32 >> U32(15)) | U32(0x100));// RGBEの指数部(9)
    I32 expo_ru_5i = I32(expo_ru_32)-126;// RGBEの指数部 0 [1,30] 31 -15 -> BIT列を削ることで指数部相当分を増やせる(最大8BIT)
    I32 expo_gu_5i = I32(expo_gu_32)-126;// RGBEの指数部 0 [1,30] 31 -15 -> BIT列を削ることで指数部相当分を増やせる(最大8BIT)
    I32 expo_bu_5i = I32(expo_bu_32)-126;// RGBEの指数部 0 [1,30] 31 -15 -> BIT列を削ることで指数部相当分を増やせる(最大8BIT)
    I32 max_exp_5i = fmaxf(fmaxf(expo_ru_5i, expo_gu_5i), expo_bu_5i);// 最大の指数部を共有ビットに指定する
    I32 rel_expo_ru_5i = expo_ru_5i - max_exp_5i;// 差分
    I32 rel_expo_gu_5i = expo_gu_5i - max_exp_5i;// 差分
    I32 rel_expo_bu_5i = expo_bu_5i - max_exp_5i;// 差分
    if (sign_ru_32) { frag_ru_9u = 0; } else { frag_ru_9u >>= U32(-rel_expo_ru_5i); }// 共有ビットと固有ビットの差分分オフセットする
    if (sign_gu_32) { frag_gu_9u = 0; } else { frag_gu_9u >>= U32(-rel_expo_gu_5i); }// 共有ビットと固有ビットの差分分オフセットする
    if (sign_bu_32) { frag_bu_9u = 0; } else { frag_bu_9u >>= U32(-rel_expo_bu_5i); }// 共有ビットと固有ビットの差分分オフセットする
    return (U32(fminf(fmaxf(max_exp_5i + 15, 0), 31)) << U32(27)) | (frag_bu_9u << U32(18)) | (frag_gu_9u << U32(9)) | frag_ru_9u;
  }
  HK_INLINE Vec3  convertR9G9B9E5ToVec3 (U32  value) {
    U32 r   = U32(value & U32(0x000001FF));           // 9BIT指数部(R) -> 0.ABCDEFGHI = ABCDEFGHI * 2^-9
    U32 g   = U32(value & U32(0x0003FE00)) >> U32(9) ;// 9BIT指数部(G)
    U32 b   = U32(value & U32(0x07FC0000)) >> U32(18);// 9BIT指数部(B)
    U32 e   = U32((value& U32(0xF8000000)) >> U32(27)) /*-9-15+127*/+ U32(103);// 5BIT仮数部    -> 浮動小数へ
    F32 exp = convertU32ToF32(e << U32(23));
#if defined(HK_LANG_CUDA_CXX)
    return make_float3(F32(r) * exp, F32(g) * exp, F32(b) * exp);
#else
    return Vec3(F32(r) * exp, F32(g) * exp, F32(b) * exp);
#endif
  }
  // NORM32
  HK_INLINE U32   convertF32ToUNorm32(F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value, 0.0, 1.0);
    if (v == 1.0f) { return 0xFFFFFFFF; }
    return U32(round(v) * F32(0xFFFFFFFF)));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = saturate(value);
    if (v == 1.0f) { return 0xFFFFFFFF; }
    return U32(round(v * F32(0xFFFFFFFF)));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = __saturatef(value);
    if (v == 1.0f) { return 0xFFFFFFFF; }
    return U32(roundf(__saturatef(value)) * F32(0xFFFFFFFF)));
#endif
#if defined(HK_LANG_CXX_HOST)
    auto v = fmaxf(fminf(value, 1.0f), 0.0f);
    if (v == 1.0f) { return 0xFFFFFFFF; }
    return U32(roundf(v * F32(0xFFFFFFFF)));
#endif
  }
  HK_INLINE I32   convertF32ToSNorm32(F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value, F32(-1), F32(1));
    if (v >= 1.0f) { return I32(0x7FFFFFFF); }
    return I32(round(v * F32(0x7FFFFFFF)));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = clamp(value, F32(-1), F32(1));
    if (v >= 1.0f) { return I32(0x7FFFFFFF); }
    return I32(round(v * F32(0x7FFFFFFF)));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = fmaxf(fminf(value, +1.0f), -1.0f);
    if (v >= 1.0f) { return I32(0x7FFFFFFF); }
    return I32(roundf(v * F32(0x7FFFFFFF)));
#endif
#if defined(HK_LANG_CXX_HOST)
    auto v = fmaxf(fminf(value, +1.0f), -1.0f);
    if (v >= 1.0f) { return I32(0x7FFFFFFF); }
    return I32(roundf(v * F32(0x7FFFFFFF)));
#endif
  }
  HK_INLINE F32   convertUNorm32ToF32(U32 value) {
    if (value == 0xFFFFFFFF) { return 1.0f; }
    if (value == 0x00000000) { return 0.0f; }
    return value / F32(0xFFFFFFFE);
  }
  HK_INLINE F32   convertSNorm32ToF32(I32 value) {
    if (value == 0x7FFFFFFF) { return  1.0f; }
    if (value == 0x00000000) { return  0.0f; }
    if (value == 0x80000000) { return -1.0f; }
    if (value == 0x80000001) { return -1.0f; }
    return value / F32(0x7FFFFFFF);
  }
  // NORM16
  HK_INLINE U32   convertF32ToUNorm16(F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value,0.0,1.0);
    if (v >= 1.0) { return U32(0xFFFF); }
    return U32(round(v * 65535.0));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = saturate(value);
    if (v >= 1.0) { return U32(0xFFFF); }
    return U32(round(v * 65535.0));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = __saturatef(value);
    if (v >= 1.0f) { return U32(0xFFFF); }
    return U32(roundf(v * 65535.0f));
#endif
#if defined(HK_LANG_CXX_HOST)
    F32 v = fmaxf(fminf(value, 1.0f), 0.0f);
    if (v >= 1.0f) { return U32(0xFFFF); }
    return U32(roundf(v * 65535.0f));
#endif
  }
  HK_INLINE U32   convertF32ToSNorm16(F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value, -1.0, 1.0);
    if (v == 1.0f) { return 0x7FFF; }
    return convertI16ToU32(round(v * 32767.0));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = clamp(value, -1.0, 1.0);
    if (v == 1.0f) { return 0x7FFF; }
    return convertI16ToU32(round(v * 32767.0));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = fminf(fmaxf(value, -1.0f), 1.0f);
    if (v == 1.0f) { return 0x7FFF; }
    return convertI16ToU32(roundf(v * 32767.f));
#endif
#if defined(HK_LANG_CXX_HOST)
    F32 v = fminf(fmaxf(value, -1.0f), 1.0f);
    if (v == 1.0f) { return U32(0x7FFF); }
    return convertI16ToU32(roundf(v * 32767.f));
#endif
  }
  HK_INLINE F32   convertUNorm16ToF32(U32 value) {
    if (value == 0xFFFFFFFF) { return 1.0f; }
    if (value == 0x00000000) { return 0.0f; }
    return value / F32(0xFFFFFFFE);
  }
  HK_INLINE F32   convertSNorm16ToF32(U32 value) {
    if (value == U32(0x7FFF)) { return  1.0f; }
    if (value == U32(0x0000)) { return  0.0f; }
    if (value == U32(0x8000)) { return -1.0f; }
    if (value == U32(0x8001)) { return -1.0f; }
    return convertU32ToI16(value) / 32767.f;
  }
  // NORM8
  HK_INLINE U32   convertF32ToUNorm8 (F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value, 0.0, 1.0);
    if (v >= 1.0) { return U32(0xFF); }
    return U32(round(v * 255.0));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = saturate(value);
    if (v >= 1.0) { return U32(0xFF); }
    return U32(round(v * 255.0));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = __saturatef(value);
    if (v >= 1.0f) { return U32(0xFF); }
    return U32(roundf(v * 255.0f));
#endif
#if defined(HK_LANG_CXX_HOST)
    F32 v = fmaxf(fminf(value, 1.0f), 0.0f);
    if (v >= 1.0f) { return U32(0xFF); }
    return U32(roundf(v * 255.0f));
#endif
  }
  HK_INLINE I32   convertF32ToSNorm8 (F32 value) {
#if defined(HK_LANG_GLSL)
    F32 v = clamp(value,-1.0, 1.0);
    if (v >= 1.0) { return I32(0x7F); }
    return convertI8ToU32(round(v * 127.0));
#endif
#if defined(HK_LANG_HLSL)
    F32 v = clamp(value, -1.0, 1.0);
    if (v >= 1.0) { return I32(0x7F); }
    return convertI8ToU32(round(v * 127.0));
#endif
#if defined(HK_LANG_CUDA_CXX)
    F32 v = fmaxf(fminf(value, 1.0f),-1.0f);
    if (v >= 1.0f) { return U32(0x7F); }
    return convertI8ToU32(roundf(v * 127.0f));
#endif
#if defined(HK_LANG_CXX_HOST)
    F32 v = fmaxf(fminf(value, 1.0f), -1.0f);
    if (v >= 1.0f) { return U32(0x7F); }
    return convertI8ToU32(roundf(v * 127.0f));
#endif
  }
  HK_INLINE F32   convertUNorm8ToF32(U32 value) {
    if (value == U32(0xFF)) { return 1.0f; }
    if (value == U32(0x00)) { return 0.0f; }
    return value / F32(0xFE);
  }
  HK_INLINE F32   convertSNorm8ToF32(U32 value) {
    if (value == U32(0x7F)) { return  1.0f; }
    if (value == U32(0x00)) { return  0.0f; }
    if (value == U32(0x80)) { return -1.0f; }
    if (value == U32(0x81)) { return -1.0f; }
    return convertU32ToI8(value) / F32(0x7F);
  }
  HK_INLINE U32   convertVec2ToUNorm16X2(Vec2 value){
#if defined(HK_LANG_GLSL)
    return packUnorm2x16(value);
#else
    U32 ux = convertF32ToUNorm16(value.x);
    U32 uy = convertF32ToUNorm16(value.y) << U32(16);
    return ux | uy;
#endif
  }
  HK_INLINE U32   convertVec2ToSNorm16X2(Vec2 value){
#if defined(HK_LANG_GLSL)
    return packSnorm2x16(value);
#else
    U32 ux = convertF32ToSNorm16(value.x);
    U32 uy = convertF32ToSNorm16(value.y) << U32(16);
    return ux | uy;
#endif
  }
  HK_INLINE U32   convertVec2ToUNorm8X4(Vec4 value){
#if defined(HK_LANG_GLSL)
    return packUnorm4x8(value);
#else
    U32 ux = convertF32ToUNorm8(value.x);
    U32 uy = convertF32ToUNorm8(value.y) << U32(8);
    U32 uz = convertF32ToUNorm8(value.z) << U32(16);
    U32 uw = convertF32ToUNorm8(value.w) << U32(24);
    return ux | uy | uz | uw;
#endif
  }
  HK_INLINE U32   convertVec2ToSNorm8X4(Vec4 value){
#if defined(HK_LANG_GLSL)
    return packSnorm4x8(value);
#else
    U32 ux = convertF32ToSNorm8(value.x);// 128
    U32 uy = convertF32ToSNorm8(value.y) << U32(8);
    U32 uz = convertF32ToSNorm8(value.z) << U32(16);
    U32 uw = convertF32ToSNorm8(value.w) << U32(24);
    return ux | uy | uz | uw;
#endif
  }

#if defined(HK_LANG_CXX) 
}
#endif
