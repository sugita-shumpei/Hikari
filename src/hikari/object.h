#ifndef HK_OBJECT__H
#define HK_OBJECT__H

#include "data_type.h"
#include "uuid.h"

#define HK_OBJECT_TYPEID_Unknown HK_UUID_DEFINE(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0)

#if defined(__cplusplus)
struct HKUnknown {
    static HK_INLINE HK_CXX_CONSTEXPR auto TypeID() HK_CXX_NOEXCEPT -> HKUUID { return HK_OBJECT_TYPEID_Unknown; }
    virtual HKU32  HK_API addRef()  = 0;
    virtual HKU32  HK_API release() = 0;
    virtual HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface) = 0;

    template<typename T>
    HKBool queryInterface(T** ppInterface) { return queryInterface(T::TypeID(), (T**)ppInterface); }
};
#else
typedef struct HKUnknown HKUnknown;
#endif

HK_EXTERN_C HK_DLL HKU32  HK_API HKUnknown_addRef (HKUnknown* pObj) ;
HK_EXTERN_C HK_DLL HKU32  HK_API HKUnknown_release(HKUnknown* pObj) ;
HK_EXTERN_C HK_DLL HKBool HK_API HKUnknown_queryInterface(HKUnknown* pObj,HKUUID iid, void** ppvInterface) ;

#if defined(__cplusplus)
template<typename T>
HK_INLINE HKU32 HKObject_addRef(T* pObj) { 
    if (!pObj){ return 0; }
    else { return pObj->addRef(); }
}
template<typename T>
HK_INLINE HKU32 HKObject_release(T* pObj) { 
    if (!pObj){ return 0; }
    else { return pObj->release(); }
}
template<typename T>
HK_INLINE HKU32 HKObject_queryInterface(T* pObj, HKUUID iid, void** ppvInterface) {
    if (!pObj){ return false;}
    return pObj->queryInterface(iid,ppvInterface);
}
#endif

#endif
