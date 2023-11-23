#ifndef HK_PLUGIN__H
#define HK_PLUGIN__H
#if !defined(__CUDACC__)

#include "platform.h"
#include "data_type.h"
#include "object.h"
#include "ref_ptr.h"

#define HK_OBJECT_TYPEID_Plugin        HK_UUID_DEFINE(0xe34cbdbc, 0x4446, 0x422e, 0xb3, 0xe1, 0x37, 0x7b, 0x64, 0xf7, 0x2e, 0x4d)
#define HK_OBJECT_TYPEID_PluginManager HK_UUID_DEFINE(0xbd6e8acd, 0x33c4, 0x4e09, 0xa0, 0x78, 0xbe, 0xea, 0x43, 0xad, 0x84, 0x95)
#define HK_OBJECT_TYPEID_PluginCore    HK_UUID_DEFINE(0x8fcff72f, 0xbe0c, 0x4cdd, 0x9d, 0x8, 0xae, 0x14, 0x5d, 0xa3, 0x6, 0x86)
#if defined(__cplusplus)

struct HKPlugin : public HKUnknown {
	static HK_CXX11_CONSTEXPR  HKUUID TypeID() { return HK_OBJECT_TYPEID_Plugin; }
	virtual HKUUID      HK_API getID()                        const = 0;
	virtual HKU32       HK_API getDependedCount()             const = 0;
	virtual HKUUID      HK_API getDependedID(HKU32 idx)       const = 0;
	virtual HKUnknown*  HK_API createObject(HKUUID iid)             = 0;
	virtual HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) = 0;

	template<typename T>
	HK_INLINE T*               createObject() { return (T*)createObject(T::TypeID()); }
	template<typename FunctionPtr>
	HK_INLINE FunctionPtr      getProcAddress(HKCStr name) {
		return reinterpret_cast<FunctionPtr>(internal_getProcAddress(name));
	}
};
struct HKPluginManager : public HKUnknown {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_PluginManager; }
	// プラグインを読み込む. もし同系統のプラグインがすでに読み込まれていれば, 
	// 直ちに開放する.
	virtual HKBool      HK_API load   (HKCStr  filename)      = 0;
	// プラグインがマネージャに含まれているかどうか確認
	virtual HKBool      HK_API contain(HKUUID  pluginid)const = 0;
	// プラグインを開放する, 依存関係にあるプラグインがある場合  
	// 依存関係にあるプラグインが全て解放されるまでの間保持される
	virtual void        HK_API unload (HKUUID  pluginid)      = 0;
	// 依存関係にあるプラグインの数を調べる
	virtual HKU32       HK_API getDependedCount(HKUUID pluginid)        const = 0;
	// 依存関係にあるプラグインを取得する
	virtual HKUUID      HK_API getDependedID(HKUUID pluginid,HKU32 idx) const = 0;
	// 任意のプラグインからオブジェクトを作成する
	// 依存数が多い順にcreateObjectを実行し, 生成できるまで繰り返す
	virtual HKUnknown*  HK_API createObject(HKUUID iid) = 0;
	// 特定のプラグインからオブジェクトを作成する
	virtual HKUnknown*  HK_API createObjectFromPlugin(HKUUID pluginid, HKUUID iid)  = 0;
	// 任意のプラグインから関数ポインタを作成する
	// 依存数が多い順にinternal_getProcAddressを実行し, 取得できるまで繰り返す
	virtual HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) = 0;
	// 特定のプラグインから関数ポインタを作成する
	virtual HK_PFN_PROC HK_API internal_getProcAddressFromPlugin(HKUUID pluginid, HKCStr name)  = 0;

	template<typename T>
	HK_INLINE T* createObject() { return (T*)createObject(T::TypeID()); }
	template<typename T>
	HK_INLINE T* createObjectFromPlugin(HKUUID pluginid) { return (T*)createObjectFromPlugin(pluginid,T::TypeID()); }

	template<typename FunctionPtr>
	HK_INLINE FunctionPtr      getProcAddress(HKCStr name) {
		return reinterpret_cast<FunctionPtr>(internal_getProcAddress(name));
	}
	template<typename FunctionPtr>
	HK_INLINE FunctionPtr      getProcAddressFromPlugin(HKUUID pluginid,HKCStr name) {
		return reinterpret_cast<FunctionPtr>(internal_getProcAddressFromPlugin(pluginid,name));
	}
};

#else
typedef struct HKPlugin        HKPlugin;
typedef struct HKPluginManager HKPluginManager;
#endif
HK_NAMESPACE_TYPE_ALIAS(Plugin);
HK_NAMESPACE_TYPE_ALIAS(PluginManager);

// このプラグイン関数の実行は, hikari.dll以外必要としない
// 0. プラグインマネジャーが有効なハンドルでない場合      →NULL 
// 1. 同系統のプラグインがすでに読み込まれていた場合      →NULL 
// 2. 依存関係をチェック.   依存プラグインが欠けていた場合→NULL 
// 3. そうでなければプラグインを読み取る                         
HK_EXTERN_C typedef HKPlugin* (HK_API* Pfn_HKPlugin_create)(HKPluginManager*);

HK_NAMESPACE_TYPE_ALIAS(Plugin);
HK_OBJECT_C_DERIVE_METHODS(HKPlugin);

HK_NAMESPACE_TYPE_ALIAS(PluginManager);
HK_OBJECT_C_DERIVE_METHODS(HKPluginManager);


HK_EXTERN_C HK_DLL_FUNCTION HKPluginManager* HK_DLL_FUNCTION_NAME(HKPluginManager_create)();
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKPluginManager_load)(HKPluginManager*,HKCStr);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKPluginManager_contain)(const HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKPluginManager_unload)(HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKPluginManager_getDependedCount)(const HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HKUUID           HK_DLL_FUNCTION_NAME(HKPluginManager_getDependedID)(const HKPluginManager*, HKUUID,HKU32);
HK_EXTERN_C HK_DLL_FUNCTION HKUnknown*       HK_DLL_FUNCTION_NAME(HKPluginManager_createObject)( HKPluginManager*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HKUnknown*       HK_DLL_FUNCTION_NAME(HKPluginManager_createObjectFromPlugin)(HKPluginManager*, HKUUID, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HK_PFN_PROC      HK_DLL_FUNCTION_NAME(HKPluginManager_internal_getProcAddress)(HKPluginManager*,HKCStr);
HK_EXTERN_C HK_DLL_FUNCTION HK_PFN_PROC      HK_DLL_FUNCTION_NAME(HKPluginManager_internal_getProcAddressFromPlugin)(HKPluginManager*, HKUUID, HKCStr);

HK_EXTERN_C HK_DLL_FUNCTION HKUUID           HK_DLL_FUNCTION_NAME(HKPlugin_getID)(const HKPlugin*);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKPlugin_getDependedCount)(const HKPlugin*);
HK_EXTERN_C HK_DLL_FUNCTION HKUUID           HK_DLL_FUNCTION_NAME(HKPlugin_getDependedID)(const HKPlugin*,HKU32);
HK_EXTERN_C HK_DLL_FUNCTION HKUnknown*       HK_DLL_FUNCTION_NAME(HKPlugin_createObject)(HKPlugin*, HKUUID);
HK_EXTERN_C HK_DLL_FUNCTION HK_PFN_PROC      HK_DLL_FUNCTION_NAME(HKPlugin_internal_getProcAddress)(HKPlugin*, HKCStr);

#if defined(__cplusplus)
HK_OBJECT_CREATE_TRAITS(HKPluginManager);
#endif

#endif
#endif
