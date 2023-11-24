#ifndef HK_CORE_PLUGIN__H
#define HK_CORE_PLUGIN__H
#if !defined(__CUDACC__)
// TODO �O����DLL���������@

#include <hikari/platform.h>
#include <hikari/data_type.h>
#include <hikari/object.h>
#include <hikari/ref_ptr.h>

#define HK_OBJECT_TYPEID_Plugin        HK_UUID_DEFINE(0xe34cbdbc, 0x4446, 0x422e, 0xb3, 0xe1, 0x37, 0x7b, 0x64, 0xf7, 0x2e, 0x4d)
#define HK_OBJECT_TYPEID_PluginManager HK_UUID_DEFINE(0xbd6e8acd, 0x33c4, 0x4e09, 0xa0, 0x78, 0xbe, 0xea, 0x43, 0xad, 0x84, 0x95)
#define HK_OBJECT_TYPEID_PluginCore    HK_UUID_DEFINE(0x8fcff72f, 0xbe0c, 0x4cdd, 0x9d, 0x8 , 0xae, 0x14, 0x5d, 0xa3, 0x6 , 0x86)

#define HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME, FUNC) \
do { if (NAME == #FUNC) { return reinterpret_cast<HK_PFN_PROC>(FUNC); } } while(false)

#if defined(__cplusplus)

struct HKPlugin : public HKUnknown {
	static HK_CXX11_CONSTEXPR  HKUUID TypeID() { return HK_OBJECT_TYPEID_Plugin; }
	virtual HKUUID      HK_API getID()                        const = 0;
	virtual HKU32       HK_API getDependedCount()             const = 0;
	virtual HKUUID      HK_API getDependedID(HKU32 idx)       const = 0;
	virtual HKUnknown*  HK_API createObject(HKUUID iid)             = 0;
	// virtual HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) = 0;

	template<typename T>
	HK_INLINE T*               createObject() { return (T*)createObject(T::TypeID()); }
	//template<typename FunctionPtr>
	//HK_INLINE FunctionPtr      getProcAddress(HKCStr name) {
	//	return reinterpret_cast<FunctionPtr>(internal_getProcAddress(name));
	//}
};
struct HKPluginManager : public HKUnknown {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_PluginManager; }
	// �v���O�C����ǂݍ���. �������n���̃v���O�C�������łɓǂݍ��܂�Ă����, 
	// �����ɊJ������.
	virtual HKBool      HK_API load   (HKCStr  filename)      = 0;
	// �v���O�C�����}�l�[�W���Ɋ܂܂�Ă��邩�ǂ����m�F
	virtual HKBool      HK_API contain(HKUUID  pluginid)const = 0;
	// �v���O�C�����J������, �ˑ��֌W�ɂ���v���O�C��������ꍇ  
	// �ˑ��֌W�ɂ���v���O�C�����S�ĉ�������܂ł̊ԕێ������
	virtual void        HK_API unload (HKUUID  pluginid)      = 0;
	// �ˑ��֌W�ɂ���v���O�C���̐��𒲂ׂ�
	virtual HKU32       HK_API getDependedCount(HKUUID pluginid)        const = 0;
	// �ˑ��֌W�ɂ���v���O�C�����擾����
	virtual HKUUID      HK_API getDependedID(HKUUID pluginid,HKU32 idx) const = 0;
	// �C�ӂ̃v���O�C������I�u�W�F�N�g���쐬����
	// �ˑ�������������createObject�����s��, �����ł���܂ŌJ��Ԃ�
	virtual HKUnknown*  HK_API createObject(HKUUID iid) = 0;
	// ����̃v���O�C������I�u�W�F�N�g���쐬����
	virtual HKUnknown*  HK_API createObjectFromPlugin(HKUUID pluginid, HKUUID iid)  = 0;
	// �C�ӂ̃v���O�C������֐��|�C���^���쐬����
	// �ˑ�������������internal_getProcAddress�����s��, �擾�ł���܂ŌJ��Ԃ�
	virtual HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) = 0;
	// ����̃v���O�C������֐��|�C���^���쐬����
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
#define HK_PLUGIN_MANAGER_GET_PROC_ADDRESS(PLUGIN_MANAGER, FUNCTION) \
	PLUGIN_MANAGER->getProcAddress<Pfn##_##FUNCTION>(#FUNCTION)
#define HK_PLUGIN_MANAGER_INIT_FUNCTION(PLUGIN_MANAGER, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_PLUGIN_MANAGER_GET_PROC_ADDRESS(PLUGIN_MANAGER, FUNCTION)
#define HK_PLUGIN_MANAGER_GET_PROC_ADDRESS_FROM_PLUGIN(PLUGIN_MANAGER, UUID, FUNCTION) \
	PLUGIN_MANAGER->getProcAddressFromPlugin<Pfn##_##FUNCTION>(UUID,#FUNCTION)
#define HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(PLUGIN_MANAGER, UUID, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_PLUGIN_MANAGER_GET_PROC_ADDRESS_FROM_PLUGIN(PLUGIN_MANAGER, UUID,  FUNCTION)
#else
typedef struct HKPlugin        HKPlugin;
typedef struct HKPluginManager HKPluginManager; 
#define HK_PLUGIN_MANAGER_GET_PROC_ADDRESS(PLUGIN_MANAGER, FUNCTION) \
	(Pfn_##FUNCTION)HKPluginManager_internal_getProcAddress(PLUGIN_MANAGER,#FUNCTION)
#define HK_PLUGIN_MANAGER_INIT_FUNCTION(PLUGIN_MANAGER, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_PLUGIN_MANAGER_GET_PROC_ADDRESS(PLUGIN_MANAGER, FUNCTION)
#define HK_PLUGIN_MANAGER_GET_PROC_ADDRESS_FROM_PLUGIN(PLUGIN_MANAGER, UUID, FUNCTION) \
	(Pfn_##FUNCTION)HKPluginManager_internal_getProcAddressFromPlugin(PLUGIN_MANAGER,UUID,#FUNCTION)
#define HK_PLUGIN_MANAGER_INIT_FUNCTION_FROM_PLUGIN(PLUGIN_MANAGER, UUID, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_PLUGIN_MANAGER_GET_PROC_ADDRESS_FROM_PLUGIN(PLUGIN_MANAGER,UUID, FUNCTION)
#endif
HK_NAMESPACE_TYPE_ALIAS(Plugin);
HK_NAMESPACE_TYPE_ALIAS(PluginManager);

// ���̃v���O�C���֐��̎��s��, hikari.dll�ȊO�K�v�Ƃ��Ȃ�
// 0. �v���O�C���}�l�W���[���L���ȃn���h���łȂ��ꍇ      ��NULL 
// 1. ���n���̃v���O�C�������łɓǂݍ��܂�Ă����ꍇ      ��NULL 
// 2. �ˑ��֌W���`�F�b�N.   �ˑ��v���O�C���������Ă����ꍇ��NULL 
// 3. �����łȂ���΃v���O�C����ǂݎ��                         
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
//HK_EXTERN_C HK_DLL_FUNCTION HK_PFN_PROC      HK_DLL_FUNCTION_NAME(HKPlugin_internal_getProcAddress)(HKPlugin*, HKCStr);

#if defined(__cplusplus)
HK_OBJECT_CREATE_TRAITS(HKPluginManager);
#endif

#endif
#endif
