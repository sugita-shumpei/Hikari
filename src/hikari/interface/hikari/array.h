#ifndef HK_ARRAY__H
#define HK_ARRAY__H
#include <hikari/data_type.h>
#include <hikari/object.h>
#include "math/vec.h"
#include "math/matrix.h"

#define HK_ARRAY_DEFINE_COMMON(TYPE) \
	HK_EXTERN_C HK_DLL HKArray##TYPE*  HK_API HKArray##TYPE##_create  ();                             \
	HK_EXTERN_C HK_DLL HKArray##TYPE*  HK_API HKArray##TYPE##_clone   (const HKArray##TYPE*   p);     \
	HK_EXTERN_C HK_DLL HKU32           HK_API HKArray##TYPE##_getCount(const HKArray##TYPE*   p);     \
	HK_EXTERN_C HK_DLL void            HK_API HKArray##TYPE##_setCount(HKArray##TYPE* p,HKU32 c);     \
	HK_EXTERN_C HK_DLL void            HK_API HKArray##TYPE##_setValue(HKArray##TYPE* p,HKU32 idx, HK##TYPE  v); \
	HK_EXTERN_C HK_DLL HK##TYPE        HK_API HKArray##TYPE##_getValue(const HKArray##TYPE* p,HKU32 idx);       \
	HK_EXTERN_C HK_DLL const HK##TYPE* HK_API HKArray##TYPE##_internal_getPointer_const(const HKArray##TYPE*   p); \
	HK_EXTERN_C HK_DLL       HK##TYPE* HK_API HKArray##TYPE##_internal_getPointer(HKArray##TYPE* p); \
	HK_INLINE HKBool HKArray##TYPE##_isEmpty(const HKArray##TYPE* p){ return HKArray##TYPE##_getCount(p) > 0; } \
	HK_INLINE void   HKArray##TYPE##_clear(HKArray##TYPE* p){ return HKArray##TYPE##_setCount(p,0); } 

#if defined(__cplusplus)
#define HK_ARRAY_DEFINE(TYPE) \
	struct HKArray##TYPE : public HKUnknown { \
		static    HKArray##TYPE* create() ; \
		virtual   HKArray##TYPE* HK_API clone() const                     = 0; \
		virtual       HKU32      HK_API getCount() const                  = 0; \
		virtual       void       HK_API setCount(HKU32 count)             = 0; \
		virtual       void       HK_API setValue(HKU32 idx, HK##TYPE v)   = 0; \
		virtual       HK##TYPE   HK_API getValue(HKU32 idx) const         = 0; \
		virtual const HK##TYPE*  HK_API internal_getPointer_const() const = 0; \
		virtual       HK##TYPE*  HK_API internal_getPointer()             = 0; \
		const HK##TYPE* getPointer() const { return internal_getPointer_const(); } \
		      HK##TYPE* getPointer()       { return internal_getPointer();       } \
		      HKBool    isEmpty() const { return getCount()>0; } \
		      void      clear() { setCount(0);} \
	}; \
	HK_ARRAY_DEFINE_COMMON(TYPE); \
	HKArray##TYPE*  HKArray##TYPE::create() { return HKArray##TYPE##_create(); }
#else
#define HK_ARRAY_DEFINE(TYPE) \
	typedef struct HKArray##TYPE HKArray##TYPE; \
	HK_ARRAY_DEFINE_COMMON(TYPE)
#endif

HK_ARRAY_DEFINE(U8);
HK_ARRAY_DEFINE(U16);
HK_ARRAY_DEFINE(U32);
HK_ARRAY_DEFINE(U64);

HK_ARRAY_DEFINE(I8);
HK_ARRAY_DEFINE(I16);
HK_ARRAY_DEFINE(I32);
HK_ARRAY_DEFINE(I64);

HK_ARRAY_DEFINE(F32);
HK_ARRAY_DEFINE(F64);

HK_ARRAY_DEFINE(Byte);
HK_ARRAY_DEFINE(Char);
HK_ARRAY_DEFINE(Bool);
HK_ARRAY_DEFINE(CStr);

HK_ARRAY_DEFINE(Vec2);
HK_ARRAY_DEFINE(Vec3);
HK_ARRAY_DEFINE(Vec4);

HK_ARRAY_DEFINE(Mat2x2);
HK_ARRAY_DEFINE(Mat3x3);
HK_ARRAY_DEFINE(Mat4x4);

#endif
