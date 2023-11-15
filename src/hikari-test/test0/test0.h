#ifndef HK_TEST_TEST0__H
#define HK_TEST_TEST0__H

#include <hikari/object.h>
#include <hikari/ref_cnt_object.h>
#include <hikari/ref_ptr.h>
#ifdef __cplusplus
#include <string>
#include <cmath>
#endif

#define HK_OBJECT_TYPEID_SampleObject HK_UUID_DEFINE(0xd6718642, 0x555d, 0x44f5, 0x9c, 0xb1, 0xaa, 0x1c, 0xc9, 0x80, 0xce, 0xbe)
#if defined(__cplusplus) 
struct HK_DLL HKSampleObject : public HKUnknown, protected HKRefCntObject {
	static HK_INLINE HK_CXX_CONSTEXPR auto TypeID() HK_CXX_NOEXCEPT -> HKUUID { return HK_OBJECT_TYPEID_SampleObject; }
	HK_API HKSampleObject() HK_CXX_NOEXCEPT;
	virtual HK_API ~HKSampleObject() HK_CXX_NOEXCEPT;
	virtual HKU32  HK_API release();
	virtual HKU32  HK_API addRef ();
	virtual HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface);
	virtual void   HK_API setName(HKCStr name);
	virtual HKCStr HK_API getName()const ;
private:
	virtual void   HK_API destroyObject() override;
	std::string m_name;
};
#else
typedef struct HKSampleObject HKSampleObject;
#endif

HK_EXTERN_C HK_DLL HKSampleObject* HK_API HKSampleObject_create();
HK_EXTERN_C HK_DLL void   HK_API HKSampleObject_setName(HKSampleObject* pObj,HKCStr name);
HK_EXTERN_C HK_DLL HKCStr HK_API HKSampleObject_getName(const HKSampleObject* pObj);

#endif
