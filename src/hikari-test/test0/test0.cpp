#define HK_DLL_EXPORT
#include "test0.h"

HK_API HKSampleObject::HKSampleObject()  HK_CXX_NOEXCEPT :HKUnknown(), HKRefCntObject(), m_name{ "" } {}
HK_API HKSampleObject::~HKSampleObject() HK_CXX_NOEXCEPT {}

HKU32 HK_API HKSampleObject::release() { return HKRefCntObject::release(); }

HKU32 HK_API HKSampleObject::addRef() { return HKRefCntObject::addRef(); }

HKBool HK_API HKSampleObject::queryInterface(HKUUID iid, void** ppvInterface)
{
	if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_SampleObject) {
		addRef();
		*ppvInterface = this;
		return true;
	}
	else {
		return false;
	}
}

void HK_API HKSampleObject::setName(HKCStr name)
{
	m_name = name;
	return;
}

HKCStr HK_API HKSampleObject::getName() const
{
	return m_name.c_str();
}

void HK_API HKSampleObject::destroyObject() { return; }

HK_EXTERN_C HK_DLL HKSampleObject* HK_API HKSampleObject_create()
{
	auto ptr = new HKSampleObject();
	HKUnknown_addRef(ptr);
	return ptr;
}

HK_EXTERN_C HK_DLL void HK_API HKSampleObject_setName(HKSampleObject* pObj, HKCStr name)
{
	if (!pObj) { return; }
	pObj->setName(name);
}

HK_EXTERN_C HK_DLL HKCStr HK_API HKSampleObject_getName(const HKSampleObject* pObj)
{
	if (!pObj) { return nullptr; }
	return pObj->getName();
}
