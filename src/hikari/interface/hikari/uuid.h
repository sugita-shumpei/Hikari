#ifndef HK_UUID__H
#define HK_UUID__H

#include <hikari/data_type.h>
#include "platform.h"

typedef struct HKUUID{
    HKU32 data1; 
    HKU16 data2;
    HKU16 data3;
    HKU8  data4[8];
} HKUUID;

#if defined(__cplusplus)
#define HK_UUID_DEFINE(D1,D2,D3,D41,D42,D43,D44,D45,D46,D47,D48) HKUUID{D1,D2,D3,{D41,D42,D43,D44,D45,D46,D47,D48}}
#else
#define HK_UUID_DEFINE(D1,D2,D3,D41,D42,D43,D44,D45,D46,D47,D48) (HKUUID){D1,D2,D3,{D41,D42,D43,D44,D45,D46,D47,D48}}
#endif

HK_INLINE HK_CXX_CONSTEXPR HKBool HKUUID_equal(HKUUID v1, HKUUID v2) HK_CXX_NOEXCEPT {
    return 
    (v1.data1==v2.data1) &&
    (v1.data2==v2.data2) &&
    (v1.data3==v2.data3) &&
    (v1.data4[0]==v2.data4[0]) &&
    (v1.data4[1]==v2.data4[1]) &&
    (v1.data4[2]==v2.data4[2]) &&
    (v1.data4[3]==v2.data4[3]) &&
    (v1.data4[4]==v2.data4[4]) &&
    (v1.data4[5]==v2.data4[5]) &&
    (v1.data4[6]==v2.data4[6]) &&
    (v1.data4[7]==v2.data4[7]) ;
}
// 比較演算子
#if defined(__cplusplus)
HK_INLINE HK_CXX_CONSTEXPR HKBool operator==(const HKUUID& v1, const HKUUID& v2) HK_CXX_NOEXCEPT {
    return  HKUUID_equal(v1,v2);
}
HK_INLINE HK_CXX_CONSTEXPR HKBool operator!=(const HKUUID& v1, const HKUUID& v2) HK_CXX_NOEXCEPT {
    return !HKUUID_equal(v1,v2);
}
#endif

HK_COMPILE_TIME_ASSERT(sizeof(HKUUID)==16);

#endif
