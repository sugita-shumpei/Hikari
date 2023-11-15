#define HK_DLL_EXPORT 
#include "object.h"

HK_EXTERN_C HK_DLL HKU32  HK_API HKUnknown_addRef (HKUnknown* pObj) {
    if (!pObj){ return 0;}
    return pObj->addRef();
}

HK_EXTERN_C HK_DLL HKU32  HK_API HKUnknown_release(HKUnknown* pObj) {
    if (!pObj){ return 0;}
    return pObj->release();
}
HK_EXTERN_C HK_DLL HKBool HK_API HKUnknown_queryInterface(HKUnknown* pObj,HKUUID iid, void** ppvInterface){
    if (!pObj){ return false;}
    return pObj->queryInterface(iid,ppvInterface);
}