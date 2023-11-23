#define  HK_DLL_EXPORT
#include <hikari/plugin.h>

#include <unordered_map>
#include <vector>
#include <string>

#include <hikari/ref_cnt_object.h>
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
		if (iid == HK_OBJECT_TYPEID_ArrayShape) { return HKArrayShape_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayUnknown) { return HKArrayUnknown_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayByte) { return HKArrayByte_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayChar) { return HKArrayChar_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU8) { return HKArrayU8_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU16) { return HKArrayU16_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU32) { return HKArrayU32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU64) { return HKArrayU64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI8) { return HKArrayI8_create(); }
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
	HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) override
	{
		if (name == std::string("HKUnknown_addRef")) { return reinterpret_cast<HK_PFN_PROC>(HKUnknown_addRef); }
		if (name == std::string("HKUnknown_release")) { return reinterpret_cast<HK_PFN_PROC>(HKUnknown_release); }
		if (name == std::string("HKUnknown_queryInterface")) { return reinterpret_cast<HK_PFN_PROC>(HKUnknown_queryInterface); }
		return nullptr;
	}
	// HKPlugin ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
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
struct HK_DLL HKPluginWrapper {
	 HKPluginWrapper(HKPluginManagerImpl* manager, const std::string& filename);
	~HKPluginWrapper();

	HKPluginWrapper(const HKPluginWrapper&) = delete;
	HKPluginWrapper& operator=(const HKPluginWrapper&) = delete;
	HKPluginWrapper(HKPluginWrapper&) = delete;
	HKPluginWrapper& operator= (HKPluginWrapper&&) = delete;

	HKPlugin* getPlugin() { return m_plugin; }
	HKU32     getDependedCount() const { if (m_plugin) { return m_plugin->getDependedCount(); } else { return 0u; } }
	HKUUID    getDependedID(HKU32 idx)const { if (m_plugin) { return m_plugin->getDependedID(idx); } else { return HK_OBJECT_TYPEID_Unknown; } }
	HKUUID    getID()const { if (m_plugin) { return m_plugin->getID(); } else { return HK_OBJECT_TYPEID_Unknown; } }
private:
	HKPluginManagerImpl*   m_manager          = nullptr;
	HKDynamicLoader        m_dll              = {};
	HKPlugin*              m_plugin           = {};
	std::vector<HKPlugin*> m_depended_plugins = {};
};
struct HK_DLL HKPluginManagerImpl :public HKPluginManager, protected HKRefCntObject {
	HKPluginManagerImpl() {
		m_plugin_core = new HKPluginCoreImpl();
		m_plugin_core->addRef();
	}
	virtual     HK_API ~HKPluginManagerImpl() {
		for (auto [id,plugin] : m_plugins) {
			delete plugin;
		}
		m_plugin_core->release();
	}
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
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_PluginManager) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKBool      HK_API load(HKCStr filename) override
	{
		auto plugin_wrapper = new HKPluginWrapper(this, filename);
		if (!plugin_wrapper->getPlugin()) {
			delete plugin_wrapper;
			return false;
		}
		m_plugins.insert({ plugin_wrapper->getID(),plugin_wrapper });
		return true;
	}
	HKBool      HK_API contain(HKUUID pluginid) const override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return true; }
		return m_plugins.count(pluginid);
	}
	void        HK_API unload(HKUUID pluginid) override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return; }
		auto iter = m_plugins.find(pluginid);
		if (iter!= m_plugins.end()) {
			auto plugin_wrapper = iter->second;
			m_plugins.erase(pluginid);
			delete plugin_wrapper;
		}
	}
	HKU32       HK_API getDependedCount(HKUUID pluginid) const override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return 0; }
		auto iter = m_plugins.find(pluginid);
		if (iter != m_plugins.end()) {
			return iter->second->getDependedCount();
		}
		else {
			return 0u;
		}
	}
	HKUUID      HK_API getDependedID(HKUUID pluginid, HKU32 idx) const override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return HK_OBJECT_TYPEID_Unknown; }
		auto iter = m_plugins.find(pluginid);
		if (iter != m_plugins.end()) {
			return iter->second->getDependedID(idx);
		}
		else {
			return HK_OBJECT_TYPEID_Unknown;
		}
	}
	HKUnknown*  HK_API createObject(HKUUID iid) override
	{
		auto object = m_plugin_core->createObject(iid);
		if (object) { return object; }
		for (auto& [id, wrapper] : m_plugins) {
			object = wrapper->getPlugin()->createObject(iid);
			if (object) { return object; }
		}
		return nullptr;
	}
	HKUnknown*  HK_API createObjectFromPlugin(HKUUID pluginid, HKUUID iid) override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) {
			auto object = m_plugin_core->createObject(iid);
			if (object) { return object; }
		}
		else {
			auto iter = m_plugins.find(pluginid);
			if (iter != m_plugins.end()) {
				return iter->second->getPlugin()->createObject(iid);
			}
		}

		return nullptr;
	}
	HK_PFN_PROC HK_API internal_getProcAddress(HKCStr name) override
	{
		auto proc = m_plugin_core->internal_getProcAddress(name);
		for (auto& [id, wrapper] : m_plugins) {
			proc = wrapper->getPlugin()->internal_getProcAddress(name);
			if (proc) { return proc; }
		}
		return nullptr;
	}
	HK_PFN_PROC HK_API internal_getProcAddressFromPlugin(HKUUID pluginid, HKCStr name) override
	{
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) {
			auto proc = m_plugin_core->internal_getProcAddress(name);
			if (proc) { return proc; }
		}
		else {
			auto iter = m_plugins.find(pluginid);
			if (iter != m_plugins.end()) {
				return iter->second->getPlugin()->internal_getProcAddress(name);
			}
		}
		return nullptr;
	}

	// HKRefCntObject ÇâÓÇµÇƒåpè≥Ç≥ÇÍÇ‹ÇµÇΩ
	void HK_API destroyObject() override
	{
		return;
	}

	HKPlugin*              HK_API getPlugin(HKUUID pluginid){
		if (pluginid == HK_OBJECT_TYPEID_PluginCore) { return m_plugin_core; }
		auto iter = m_plugins.find(pluginid);
		if (iter != m_plugins.end()) {
			return iter->second->getPlugin();
		}
		return nullptr;
	}
	std::vector<HKPlugin*> HK_API getDependedPlugins(HKUUID pluginid)
	{
		auto iter = m_plugins.find(pluginid);
		if (iter != m_plugins.end()) {
			std::vector<HKPlugin*> res = {};
			res.resize(iter->second->getDependedCount());
			for (HKU32 i = 0; i < res.size(); ++i) {
				res[i] = getPlugin(iter->second->getDependedID(i));
			}
			return res;
		}
		return {};
	}
private:
	// ç≈èâÇ…ì«Ç›çûÇ‹ÇÍ, ç≈å„Ç…äJï˙Ç≥ÇÍÇÈ
	HKPlugin* m_plugin_core = nullptr;
	std::unordered_map<HKUUID, HKPluginWrapper*> m_plugins = {};
};

HKPluginWrapper:: HKPluginWrapper(HKPluginManagerImpl* manager, const std::string& filename) :m_manager{ manager }, m_dll(filename.c_str()) {
	Pfn_HKPlugin_create pfn_HKPlugin_create = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(m_dll, HKPlugin_create);
	if (pfn_HKPlugin_create && manager) {
		HKPlugin* plugin = pfn_HKPlugin_create(manager);
		if (plugin) {
			m_plugin = plugin;
			m_depended_plugins = manager->getDependedPlugins(plugin->getID());
			for (auto& plugin : m_depended_plugins) {
				if (plugin) { plugin->addRef(); }
			}
		}
	}
}
HKPluginWrapper::~HKPluginWrapper() {
	for (auto& plugin : m_depended_plugins) {
		if (plugin) { plugin->release(); }
	}
	if (m_plugin) { m_plugin->release(); }
	m_dll.reset();
}

HK_EXTERN_C HK_DLL HKPlugin*        HK_API HKPlugin_create(HKPluginManager* manager) {
	return nullptr;
}
HK_EXTERN_C HK_DLL HKPluginManager* HK_API HKPluginManager_create()
{
	auto res = new HKPluginManagerImpl();
	res->addRef();
	return res;
}
