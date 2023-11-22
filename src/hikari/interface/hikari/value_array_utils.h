#ifndef HK_VALUE_ARRAY_UTILS__H
#define HK_VALUE_ARRAY_UTILS__H
#include "object.h"


#define HK_VALUE_ARRAY_DEFINE_COMMON_DLL(TYPE)                                                                                             \
    HK_EXTERN_C HK_DLL_FUNCTION HKArray##TYPE*   HK_DLL_FUNCTION_NAME(HKArray##TYPE##_create)();                                           \
    HK_EXTERN_C HK_DLL_FUNCTION HKArray##TYPE*   HK_DLL_FUNCTION_NAME(HKArray##TYPE##_clone )(const HKArray##TYPE *p);                     \
    HK_EXTERN_C HK_DLL_FUNCTION HKU64            HK_DLL_FUNCTION_NAME(HKArray##TYPE##_getCapacity)(const HKArray##TYPE *p);                \
    HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setCapacity)(HKArray##TYPE *p, HKU64 c);             \
    HK_EXTERN_C HK_DLL_FUNCTION HKU64            HK_DLL_FUNCTION_NAME(HKArray##TYPE##_getCount)(const HKArray##TYPE *p);                   \
    HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setCount)(HKArray##TYPE *p, HKU64 c);                \
    HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setValue)(HKArray##TYPE *p, HKU64 idx, HKC##TYPE v); \
    HK_EXTERN_C HK_DLL_FUNCTION HKC##TYPE        HK_DLL_FUNCTION_NAME(HKArray##TYPE##_getValue)(const HKArray##TYPE *p, HKU64 idx);        \
    HK_EXTERN_C HK_DLL_FUNCTION const HKC##TYPE* HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getPointer_const)(const HKArray##TYPE *p);  \
    HK_EXTERN_C HK_DLL_FUNCTION HKC##TYPE*       HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getPointer)(HKArray##TYPE *p)

#if !defined(HK_RUNTIME_LOAD)
#define HK_VALUE_ARRAY_DEFINE_COMMON_INLINE(TYPE)                                                                  \
    HK_INLINE HKBool HKArray##TYPE##_isEmpty(const HKArray##TYPE *p) { return HKArray##TYPE##_getCount(p) == 0; }  \
    HK_INLINE void   HKArray##TYPE##_clear(HKArray##TYPE *p) { HKArray##TYPE##_setCount(p, 0); }
#else
#define HK_VALUE_ARRAY_DEFINE_COMMON_INLINE(TYPE)
#endif

#define HK_VALUE_ARRAY_DEFINE_COMMON(TYPE)        \
    HK_VALUE_ARRAY_DEFINE_COMMON_DLL(TYPE);       \
    HK_VALUE_ARRAY_DEFINE_COMMON_INLINE(TYPE);    \
    HK_OBJECT_C_DERIVE_METHODS(HKArray##TYPE);    \
    HK_NAMESPACE_TYPE_ALIAS(Array##TYPE)

#if defined(__cplusplus)
#define HK_VALUE_ARRAY_DEFINE(TYPE)                                                             \
	struct HKArray##TYPE : public HKUnknown                                                     \
	{                                                                                           \
		static inline HK_CXX_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_Array##TYPE; } \
		typedef HK##TYPE value_type;                                                            \
		virtual HKArray##TYPE *HK_API clone() const = 0;                                        \
		virtual HKU64 HK_API getCount() const = 0;                                              \
		virtual void HK_API setCount(HKU64 count) = 0;                                          \
		virtual HKU64 HK_API getCapacity() const = 0;                                           \
		virtual void HK_API setCapacity(HKU64 count) = 0;                                       \
		virtual void HK_API setValue(HKU64 idx, HK##TYPE v) = 0;                                \
		virtual HK##TYPE HK_API getValue(HKU64 idx) const = 0;                                  \
		virtual const HK##TYPE *HK_API internal_getPointer_const() const = 0;                   \
		virtual HK##TYPE *HK_API internal_getPointer() = 0;                                     \
		const HK##TYPE *getPointer() const { return internal_getPointer_const(); }              \
		HK##TYPE *getPointer() { return internal_getPointer(); }                                \
		HKBool isEmpty() const { return getCount() == 0; }                                      \
		HKArray##TYPE *HK_API cloneWithRef() const                                              \
		{                                                                                       \
			auto ptr = clone();                                                                 \
			ptr->addRef();                                                                      \
			return ptr;                                                                         \
		}                                                                                       \
		void clear() { setCount(0); }                                                           \
		void resize(HKU64 count) { setCount(count); }                                           \
		void reserve(HKU64 count) { setCapacity(count); }                                       \
	};                                                                                          \
	HK_VALUE_ARRAY_DEFINE_COMMON(TYPE);                                                         \
	HK_OBJECT_CREATE_TRAITS(HKArray##TYPE)
#else
#define HK_VALUE_ARRAY_DEFINE(TYPE)             \
	typedef struct HKArray##TYPE HKArray##TYPE; \
	HK_VALUE_ARRAY_DEFINE_COMMON(TYPE)
#endif


#endif
