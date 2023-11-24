#define HK_DLL_EXPORT
#include <hikari/value_array.h>
#include <hikari/ref_cnt_object.h>
#include <vector>

#define HK_ARRAY_IMPL_DEFINE_COMMON(TYPE) \
	HK_EXTERN_C HK_DLL HKArray##TYPE*   HK_API HKArray##TYPE##_create  ()                             { auto res = new HKArray##TYPE##Impl(); res->addRef(); return res; } \
	HK_EXTERN_C HK_DLL HKArray##TYPE*   HK_API HKArray##TYPE##_clone   (const HKArray##TYPE*   p)     { if (p) { return p->cloneWithRef(); } else { return nullptr; } } \
	HK_EXTERN_C HK_DLL HKU64            HK_API HKArray##TYPE##_getCount(const HKArray##TYPE*   p)     { if (p) { return p->getCount( ); } else { return 0; } } \
	HK_EXTERN_C HK_DLL void             HK_API HKArray##TYPE##_setCount(HKArray##TYPE* p,HKU64 c)     { if (p) { return p->setCount(c); } } \
	HK_EXTERN_C HK_DLL HKU64            HK_API HKArray##TYPE##_getCapacity(const HKArray##TYPE*   p) { if (p) { return p->getCapacity( ); } else { return 0; } } \
	HK_EXTERN_C HK_DLL void             HK_API HKArray##TYPE##_setCapacity(HKArray##TYPE* p,HKU64 c) { if (p) { return p->setCapacity(c); } } \
	HK_EXTERN_C HK_DLL void             HK_API HKArray##TYPE##_setValue(HKArray##TYPE* p, HKU64 idx, HKC##TYPE  v) { if (p) { return p->setValue(idx,v); } } \
	HK_EXTERN_C HK_DLL HKC##TYPE        HK_API HKArray##TYPE##_getValue(const HKArray##TYPE* p,HKU64 idx)       { if (p) { return p->getValue(idx); } else { return {}; } } \
	HK_EXTERN_C HK_DLL const HKC##TYPE* HK_API HKArray##TYPE##_internal_getPointer_const(const HKArray##TYPE*   p)  { if (p) { return ( const HKC##TYPE*)p->internal_getPointer_const(); } else { return nullptr; } } \
	HK_EXTERN_C HK_DLL       HKC##TYPE* HK_API HKArray##TYPE##_internal_getPointer(HKArray##TYPE* p) { if (p) { return (HKC##TYPE*)p->internal_getPointer(); } else { return nullptr; } }


#define HK_ARRAY_IMPL_DEFINE(TYPE) \
	struct HK_DLL HKArray##TYPE##Impl : public HKArray##TYPE, protected HKRefCntObject { \
		typedef HK##TYPE* value_type; \
		HKArray##TYPE##Impl() : HKArray##TYPE(), HKRefCntObject(), arr{}{} \
		HKArray##TYPE##Impl(const std::vector<HK##TYPE>& arr_) : HKArray##TYPE(), HKRefCntObject(), arr{arr_}{} \
		virtual  HK_API ~HKArray##TYPE##Impl() {} \
		virtual HKU32      HK_API addRef() override { return HKRefCntObject::addRef();} \
		virtual HKU32      HK_API release() override { return HKRefCntObject::release(); } \
		virtual HKBool     HK_API queryInterface(HKUUID iid, void** ppvInterface) override { \
			if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Array##TYPE) { \
				addRef(); * ppvInterface = this; \
				return true; \
			}else{ \
				return false; \
			} \
		} \
		virtual void       HK_API destroyObject() override {} \
		virtual HKArray##TYPE*   HK_API clone() const override { auto res = new HKArray##TYPE##Impl(arr); return res; } \
		virtual       HKU64      HK_API getCount() const override { return arr.size();}; \
		virtual       void       HK_API setCount(HKU64 count) override{ return arr.resize(count);}; \
		virtual       HKU64      HK_API getCapacity() const override { return arr.capacity();}; \
		virtual       void       HK_API setCapacity(HKU64 count) override{ return arr.reserve(count);}; \
		virtual       void       HK_API setValue(HKU64 idx, HK##TYPE v) override { \
			if (idx < arr.size()) { arr[idx] = v; } \
		} \
		virtual       HK##TYPE   HK_API getValue(HKU64 idx) const override { \
			if (idx < arr.size()) { return arr[idx]; } else { return {}; }  \
		} \
		virtual const HK##TYPE*  HK_API internal_getPointer_const() const override{ return arr.data();}; \
		virtual       HK##TYPE*  HK_API internal_getPointer() override { return arr.data();}; \
		std::vector<HK##TYPE> arr; \
	}; \
	HK_ARRAY_IMPL_DEFINE_COMMON(TYPE)

HK_ARRAY_IMPL_DEFINE(U8);
HK_ARRAY_IMPL_DEFINE(U16);
HK_ARRAY_IMPL_DEFINE(U32);
HK_ARRAY_IMPL_DEFINE(U64);

HK_ARRAY_IMPL_DEFINE(I8);
HK_ARRAY_IMPL_DEFINE(I16);
HK_ARRAY_IMPL_DEFINE(I32);
HK_ARRAY_IMPL_DEFINE(I64);

HK_ARRAY_IMPL_DEFINE(F32);
HK_ARRAY_IMPL_DEFINE(F64);

HK_ARRAY_IMPL_DEFINE(Byte);
HK_ARRAY_IMPL_DEFINE(Char);

HK_ARRAY_IMPL_DEFINE(Vec2);
HK_ARRAY_IMPL_DEFINE(Vec3);
HK_ARRAY_IMPL_DEFINE(Vec4);

HK_ARRAY_IMPL_DEFINE(Mat2x2);
HK_ARRAY_IMPL_DEFINE(Mat3x3);
HK_ARRAY_IMPL_DEFINE(Mat4x4);

HK_ARRAY_IMPL_DEFINE(Color);
HK_ARRAY_IMPL_DEFINE(Color8);