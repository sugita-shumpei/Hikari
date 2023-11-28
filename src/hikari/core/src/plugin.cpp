#define  HK_DLL_EXPORT
#include <hikari/plugin.h>

#include <unordered_map>
#include <vector>
#include <string>

#include <hikari/ref_cnt_object.h>
#include <hikari/ref_ptr.h>
#include <hikari/object_array.h>
#include <hikari/value_array.h>
#include <hikari/shape.h>
#include <hikari/dynamic_loader.h>
#include <hikari/stl_utils.h>

struct HK_DLL HKPluginCoreImpl : public HKPlugin, protected HKRefCntObject {
	HKU32       HK_API addRef()  override
	{
		return HKRefCntObject::addRef();
	}
	HKU32       HK_API release() override
	{
		return HKRefCntObject::release();
	}
	HKBool      HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin || iid == HK_OBJECT_TYPEID_PluginCore)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown* HK_API createObject(HKUUID iid)  override
	{
		if (iid == HK_OBJECT_TYPEID_ArrayShape)   { return HKArrayShape_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayUnknown) { return HKArrayUnknown_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayByte) { return HKArrayByte_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayChar) { return HKArrayChar_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU8)   { return HKArrayU8_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU16) { return HKArrayU16_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU32) { return HKArrayU32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU64) { return HKArrayU64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI8)  { return HKArrayI8_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI16) { return HKArrayI16_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI32) { return HKArrayI32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI64) { return HKArrayI64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayF32) { return HKArrayF32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayF64) { return HKArrayF64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec2) { return HKArrayVec2_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec3) { return HKArrayVec3_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec4) { return HKArrayVec4_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat2x2) { return HKArrayMat2x2_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat3x3) { return HKArrayMat3x3_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat4x4) { return HKArrayMat4x4_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayColor) { return HKArrayColor_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayColor8) { return HKArrayColor8_create(); }

		return nullptr;
	}
	void        HK_API destroyObject()     override
	{
		return;
	}
	//HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) override
	//{
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKUnknown_addRef);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKUnknown_release);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKUnknown_queryInterface);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPlugin_createObject);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPlugin_getID);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPlugin_getDependedCount);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPlugin_getDependedID);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPlugin_internal_getProcAddress);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_load);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_unload);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_contain);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_createObject);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_createObjectFromPlugin);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_getDependedCount);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_getDependedID);
	//	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), HKPluginManager_internal_getProcAddressFromPlugin);
	//	HK_OBJECT_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Unknown);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), U8);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), U16);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), U32);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), U64);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), I8);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), I16);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), I32);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), I64);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), F32);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), F64);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Byte);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Char);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Vec2);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Vec3);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Vec4);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Mat2x2);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Mat3x3);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Mat4x4);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Color);
	//	HK_VALUE_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(std::string(name), Color8);
	//	return nullptr;
	//}
	// HKPlugin を介して継承されました
	HKUUID      HK_API getID() const override
	{
		return HK_OBJECT_TYPEID_PluginCore;
	}
	HKU32       HK_API getDependedCount() const override
	{
		return 0;
	}
	HKUUID      HK_API getDependedID(HKU32 idx) const override
	{
		return HK_OBJECT_TYPEID_Unknown;
	}
};
struct HKPluginManagerImpl;
struct HK_DLL HKPluginWrapper : public HKUnknown , protected HKRefCntObject{
	HKPluginWrapper(HKPluginManagerImpl* manager,HKCStr filename);
	virtual HK_API ~HKPluginWrapper() {}

	void   HK_API destroyObject() override
	{
		// まず自分自身のDLLを開放
		if (m_plugin) {
			printf("%s unload\n", m_filename.c_str());
			m_plugin->release();
			m_plugin = nullptr;
		}
		m_dll.reset();
		// 次に依存ライブラリを開放
		for (auto& wrapper : m_depended_wrappers) { wrapper->release(); }
		m_depended_wrappers.clear();
		return;
	}
	HKU32  HK_API addRef() override
	{
		return HKRefCntObject::addRef();
	}
	HKU32  HK_API release() override
	{
		return HKRefCntObject::release();
	}
	HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		else {
			return false;
		}
	}

	HKDynamicLoader               m_dll               = {};
	std::string                   m_filename          = "";
	HKPlugin*                     m_plugin            = nullptr;
	std::vector<HKPluginWrapper*> m_depended_wrappers = {};
};

HK_INLINE HKPluginWrapper* HKPluginWrapper_create(HKPluginManagerImpl* manager, HKCStr filename) {
	HKPluginWrapper* res = new HKPluginWrapper(manager, filename);
	res->addRef();
	return res;
}

struct HK_DLL HKPluginManagerImpl : public HKPluginManager, protected HKRefCntObject {
	// HKPluginManager を介して継承されました
	HKPluginManagerImpl() :m_plugin_core{}, m_plugins{} {
		m_plugin_core = new HKPluginCoreImpl();
		m_plugin_core->addRef();
	}
	virtual HK_API ~HKPluginManagerImpl() {

	}
	HKU32 HK_API addRef() override
	{
		return HKRefCntObject::addRef();
	}
	HKU32 HK_API release() override
	{
		return HKRefCntObject::release();
	}
	HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_PluginManager) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		else {
			return false;
		}
	}
	HKBool HK_API load(HKCStr filename) override
	{
		auto res = HKPluginWrapper_create(this,filename);
		if (!res->m_plugin) {
			res->release(); 
			return false;
		}
		m_plugins.insert({ res->m_plugin->getID(),res });
		return true;
	}
	HKBool HK_API contain(HKUUID pluginid) const override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return true;  }
		return m_plugins.count(pluginid) > 0;
	}
	void   HK_API unload(HKUUID pluginid) override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return; }
		auto iter = m_plugins.find(pluginid);
		if (iter != std::end(m_plugins)) {
			iter->second->release();
			m_plugins.erase(pluginid);
		}
	}
	HKU32  HK_API getDependedCount(HKUUID pluginid) const override
	{
		auto iter = m_plugins.find(pluginid);
		if (iter != std::end(m_plugins)) {
			return iter->second->m_plugin->getDependedCount();
		}
		else {
			return 0u;
		}
	}
	HKUUID HK_API getDependedID(HKUUID pluginid, HKU32 idx) const override
	{
		auto iter = m_plugins.find(pluginid);
		if (iter != std::end(m_plugins)) {
			return iter->second->m_plugin->getDependedID(idx);
		}
		else {
			return HK_OBJECT_TYPEID_Unknown;
		}
	}
	HKUnknown* HK_API createObject(HKUUID iid) override
	{
		if (m_plugin_core) {
			auto res = m_plugin_core->createObject(iid);
			if (res) { return res; }
		}
		for (auto& [uuid, plugin] : m_plugins) {
			auto res = plugin->m_plugin->createObject(iid);
			if (res) { return res; }
		}
		return nullptr;
	}
	HKUnknown* HK_API createObjectFromPlugin(HKUUID pluginid, HKUUID iid) override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) {
			return m_plugin_core->createObject(iid);
		}
		auto iter = m_plugins.find(pluginid);
		if (iter != std::end(m_plugins)) {
			return iter->second->m_plugin->createObject(iid);
		}
		else {
			return nullptr;
		}
	}
	HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) override
	{
		for (auto& [uuid, plugin_wrapper] : m_plugins) {
			auto res = plugin_wrapper->m_dll.internal_getProcAddress(name);
			if (res) { return res; }
		}
		return nullptr;
	}
	HK_PFN_PROC HK_API internal_getProcAddressFromPlugin(HKUUID pluginid, HKCStr name) override
	{
		auto iter = m_plugins.find(pluginid);
		if (iter != std::end(m_plugins)) {
			return iter->second->m_dll.internal_getProcAddress(name);
		}
		else {
			return nullptr;
		}
	}
	void HK_API destroyObject() override
	{
		{
			// 
			for (auto& [iid, wrapper] : m_plugins) {
				wrapper->release();
			}
			m_plugins.clear();
		}
		if (m_plugin_core) {
			m_plugin_core->release();
			m_plugin_core = nullptr;
		}
		return;
	}

	HKPlugin*                                    m_plugin_core  = nullptr;
	std::unordered_map<HKUUID, HKPluginWrapper*> m_plugins      = {};

};

HKPluginWrapper::HKPluginWrapper(HKPluginManagerImpl* manager, HKCStr filename) :
	m_filename { filename },
	m_dll{ filename }, 
	m_plugin{ nullptr },
	m_depended_wrappers{}
{
	Pfn_HKPlugin_create pfn_HKPlugin_create = m_dll.getProcAddress<Pfn_HKPlugin_create>("HKPlugin_create");
	if (!pfn_HKPlugin_create) { return; }
	m_plugin = pfn_HKPlugin_create(manager);
	if (!m_plugin) { return;  }
	m_depended_wrappers.resize(m_plugin->getDependedCount());
	for (auto i = 0; i < m_depended_wrappers.size(); ++i) {
		m_depended_wrappers[i] = manager->m_plugins.at(m_plugin->getDependedID(i));
		m_depended_wrappers[i]->addRef();
	}
}

HK_EXTERN_C HK_DLL HKPlugin*        HK_API HKPlugin_create(HKPluginManager* manager) {
	return nullptr;
}
HK_EXTERN_C HK_DLL HKUUID           HK_API HKPlugin_getID(const HKPlugin* pl) {
	if (pl) { return pl->getID(); }
	else { return HK_OBJECT_TYPEID_Unknown; }
 }
HK_EXTERN_C HK_DLL HKU32            HK_API HKPlugin_getDependedCount(const HKPlugin* pl) {
	if (pl) { return pl->getDependedCount(); }
	else { return 0;  }
}
HK_EXTERN_C HK_DLL HKUUID           HK_API HKPlugin_getDependedID(const HKPlugin* pl, HKU32 idx) {
	if (pl) { return pl->getDependedID(idx); }
	else { return HK_OBJECT_TYPEID_Unknown; }
}
HK_EXTERN_C HK_DLL HKUnknown*       HK_API HKPlugin_createObject(HKPlugin* pl, HKUUID uuid){
	if (pl) { return pl->createObject(uuid); }
	else { return nullptr; }
}

HK_EXTERN_C HK_DLL HKPluginManager* HK_API HKPluginManager_create()
{
	auto res = new HKPluginManagerImpl();
	res->addRef();
	return res;
}
HK_EXTERN_C HK_DLL HKBool           HK_API HKPluginManager_load(HKPluginManager* pl, HKCStr name) {
	if (pl) { return pl->load(name); }
	else { return false; }
}
HK_EXTERN_C HK_DLL HKBool           HK_API HKPluginManager_contain(const HKPluginManager* pl, HKUUID uuid) {
	if (pl) { return pl->contain(uuid); }
	else { return false; }
}
HK_EXTERN_C HK_DLL void             HK_API HKPluginManager_unload(HKPluginManager* pl, HKUUID uuid) {
	if (pl) { return pl->unload(uuid); }
}
HK_EXTERN_C HK_DLL HKU32            HK_API HKPluginManager_getDependedCount(const HKPluginManager* pl, HKUUID uuid) {
	if (pl) { return pl->getDependedCount(uuid); }
	else { return 0; }
}
HK_EXTERN_C HK_DLL HKUUID           HK_API HKPluginManager_getDependedID(const HKPluginManager* pl, HKUUID uuid, HKU32 idx) {
	if (pl) { return pl->getDependedID(uuid,idx); }
	else { return HK_OBJECT_TYPEID_Unknown; }
}
HK_EXTERN_C HK_DLL HKUnknown*       HK_API HKPluginManager_createObject(HKPluginManager* pl, HKUUID uuid) {
	if (pl) { return pl->createObject(uuid); }
	else { return nullptr; }

}
HK_EXTERN_C HK_DLL HKUnknown*       HK_API HKPluginManager_createObjectFromPlugin(HKPluginManager* pl, HKUUID uuid, HKUUID objuuid) {
	if (pl) { return pl->createObjectFromPlugin(uuid,objuuid); }
	else { return nullptr; }

}
HK_EXTERN_C HK_DLL HK_PFN_PROC      HK_API HKPluginManager_internal_getProcAddress(HKPluginManager* pl, HKCStr name) {
	if (pl) { return pl->internal_getProcAddress(name); }
	else { return nullptr; }
}
HK_EXTERN_C HK_DLL HK_PFN_PROC      HK_API HKPluginManager_internal_getProcAddressFromPlugin(HKPluginManager* pl, HKUUID uuid, HKCStr name) {
	if (pl) { return pl->internal_getProcAddressFromPlugin(uuid, name); }
	else { return nullptr; }
}
