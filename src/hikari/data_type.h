#ifndef HK_DATA_TYPE__H
#define HK_DATA_TYPE__H

#include "platform.h"

typedef signed char        HKI8;
typedef signed short       HKI16;
typedef signed int         HKI32;
typedef signed long long   HKI64;

typedef unsigned char      HKU8;
typedef unsigned short     HKU16;
typedef unsigned int       HKU32;
typedef unsigned long long HKU64;

typedef float         HKF32;
typedef double        HKF64;

typedef char          HKChar;
// typedef wchar_t       HKWChar;
#if defined(__cplusplus)
typedef bool          HKBool;
#else
typedef _Bool         HKBool;
#endif
typedef unsigned char HKByte;

typedef const char*   HKCStr;
typedef void*         HKVPtr;

typedef HKI8  HKCI8;
typedef HKI16 HKCI16;
typedef HKI32 HKCI32;
typedef HKI64 HKCI64;

typedef HKU8  HKCU8;
typedef HKU16 HKCU16;
typedef HKU32 HKCU32;
typedef HKU64 HKCU64;

typedef HKF32 HKCF32;
typedef HKF64 HKCF64;

typedef HKChar HKCChar;
typedef HKByte HKCByte;
typedef HKBool HKCBool;

//   Signed Int
HK_COMPILE_TIME_ASSERT(sizeof(HKI8 )==1);
HK_COMPILE_TIME_ASSERT(sizeof(HKI16)==2);
HK_COMPILE_TIME_ASSERT(sizeof(HKI32)==4);
HK_COMPILE_TIME_ASSERT(sizeof(HKI64)==8);
// Unsigned Int
HK_COMPILE_TIME_ASSERT(sizeof(HKU8 )==1);
HK_COMPILE_TIME_ASSERT(sizeof(HKU16)==2);
HK_COMPILE_TIME_ASSERT(sizeof(HKU32)==4);
HK_COMPILE_TIME_ASSERT(sizeof(HKU64)==8);
// Floating Point
HK_COMPILE_TIME_ASSERT(sizeof(HKF32)==4);
HK_COMPILE_TIME_ASSERT(sizeof(HKF64)==8);
// Byte
HK_COMPILE_TIME_ASSERT(sizeof(HKChar)==1);
HK_COMPILE_TIME_ASSERT(sizeof(HKBool)==1);
HK_COMPILE_TIME_ASSERT(sizeof(HKByte)==1);

#endif
