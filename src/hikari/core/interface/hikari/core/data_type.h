#ifndef HK_CORE_DATA_TYPE__H
#define HK_CORE_DATA_TYPE__H
// 
typedef signed char        HKI8 ;
typedef signed short       HKI16;
typedef signed int         HKI32;
typedef signed long long   HKI64;
// 
typedef unsigned char      HKU8;
typedef unsigned short     HKU16;
typedef unsigned int       HKU32;
typedef unsigned long long HKU64;
// 
typedef float              HKF32;
typedef double             HKF64;
//
typedef unsigned char      HKByte;
typedef char               HKChar;
// 
#if !defined(__cplusplus)
typedef _Bool              HKBool;
#else
typedef  bool              HKBool;
#endif
//
typedef const char*        HKCStr;
typedef void*              HKVPtr;
// 
#endif
