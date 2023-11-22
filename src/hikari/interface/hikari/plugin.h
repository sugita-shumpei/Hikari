#ifndef HK_PLUGIN__H
#define HK_PLUGIN__H
#if !defined(__CUDACC__)

#include "platform.h"
#include "data_type.h"
#include "object.h"
#include "ref_ptr.h"

#define HK_OBJETT_TYPEID_CorePlugin    HK_UUID_DEFINE(0x8fcff72f, 0xbe0c, 0x4cdd, 0x9d, 0x8, 0xae, 0x14, 0x5d, 0xa3, 0x6, 0x86)
#define HK_OBJECT_TYPEID_Plugin        HK_UUID_DEFINE(0xe34cbdbc, 0x4446, 0x422e, 0xb3, 0xe1, 0x37, 0x7b, 0x64, 0xf7, 0x2e, 0x4d)
#define HK_OBJECT_TYPEID_PluginManager HK_UUID_DEFINE(0xbd6e8acd, 0x33c4, 0x4e09, 0xa0, 0x78, 0xbe, 0xea, 0x43, 0xad, 0x84, 0x95)

#if defined(__cplusplus)
struct HKPlugin : public HKUnknown {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_Plugin; }
	virtual HKUUID      HK_API getPluginID() const                  = 0;
	virtual HKCStr      HK_API getFilename() const                  = 0;
	virtual HKUnknown*  HK_API createObject(HKUUID iid) const       = 0;
	virtual HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) = 0;

	template<typename T>
	HK_INLINE T* createObject() { return (T*)createObject(T::TypeID()); }
	template<typename FunctionPtr>
	HK_INLINE FunctionPtr getProcAddress(HKCStr name) {
		return reinterpret_cast<FunctionPtr>(internal_getProcAddress(name));
	}
};
struct HKPluginManager : public HKUnknown {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_PluginManager; }
	virtual HKPlugin*   HK_API load(HKCStr  filename)         = 0;
	virtual HKBool      HK_API contain(HKUUID  pluginid)const = 0;
	virtual HKPlugin*   HK_API get(HKUUID  pluginid)          = 0;
	virtual void        HK_API unload(HKUUID  pluginid)       = 0;
};
#else
typedef struct HKPlugin        HKPlugin;
typedef struct HKPluginManager HKPluginManager;
#endif

HK_EXTERN_C typedef HKPlugin* (HK_API* Pfn_HKPlugin_create)();

HK_NAMESPACE_TYPE_ALIAS(Plugin);
HK_OBJECT_C_DERIVE_METHODS(HKPlugin);

HK_NAMESPACE_TYPE_ALIAS(PluginManager);
HK_OBJECT_C_DERIVE_METHODS(HKPluginManager);

HK_EXTERN_C HK_DLL_FUNCTION HKPluginManager* HK_DLL_FUNCTION_NAME(HKPluginManager_create)();
HK_EXTERN_C HK_DLL_FUNCTION HKPlugin*        HK_DLL_FUNCTION_NAME(HKPluginManager_load)(HKPluginManager*,HKCStr);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKPluginManager_contain)(const HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HKPlugin*        HK_DLL_FUNCTION_NAME(HKPluginManager_get)(const HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKPluginManager_unload)(HKPluginManager*, HKUUID);

HK_EXTERN_C HK_DLL_FUNCTION HKUUID           HK_DLL_FUNCTION_NAME(HKPlugin_getPluginID)(const HKPlugin*);
HK_EXTERN_C HK_DLL_FUNCTION HKCStr           HK_DLL_FUNCTION_NAME(HKPlugin_getFilename)(const HKPlugin*);
HK_EXTERN_C HK_DLL_FUNCTION HKUnknown*       HK_DLL_FUNCTION_NAME(HKPlugin_createObject)(HKPlugin*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HK_PFN_PROC      HK_DLL_FUNCTION_NAME(HKPlugin_internal_getProcAddress)(HKPlugin*, HKCStr);

#endif
#endif
