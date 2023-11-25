#define HK_DLL_EXPORT
#define VK_NO_PROTOTYPES
#include <hikari/graphics/vulkan/plugin.h>
#include <hikari/graphics/vulkan/entry.h>
#include <hikari/graphics/vulkan/entry_impl.h>
#include <hikari/graphics/vulkan/instance.h>
#include <hikari/ref_cnt_object.h>
#include <memory>
struct HK_DLL HKPluginGraphicsVulkanImpl : public HKPlugin, protected HKRefCntObject {
	HKPluginGraphicsVulkanImpl() noexcept : HKPlugin{}, HKRefCntObject{} {}
	virtual ~HKPluginGraphicsVulkanImpl() {}

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
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin || iid == HK_OBJECT_TYPEID_PluginGraphicsVulkan)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown*  HK_API createObject(HKUUID iid)  override
	{
		if (iid == HK_OBJECT_TYPEID_GraphicsEntry) {
			return HKGraphicsEntry_create();
		}
		if (iid == HK_OBJECT_TYPEID_GraphicsVulkanEntry) {
			return HKGraphicsEntry_create();
		}
		return nullptr;
	}
	void        HK_API destroyObject()     override
	{
		return;
	}
	// HKPlugin ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
	HKUUID      HK_API getID() const override
	{
		return HK_OBJECT_TYPEID_PluginGraphicsVulkan;
	}
	HKU32       HK_API getDependedCount() const override
	{
		return 1;
	}
	HKUUID      HK_API getDependedID(HKU32 idx) const override
	{
		if (idx == 0) {
			return HK_OBJECT_TYPEID_PluginGraphicsCommon;
		}
		return HK_OBJECT_TYPEID_Unknown;
	}
};

HK_EXTERN_C HK_DLL HKPlugin* HK_API HKPlugin_create(HKPluginManager* manager) {
	if (!manager) { return nullptr; }
	if (!manager->contain(HK_OBJECT_TYPEID_PluginGraphicsCommon)) { return nullptr; }
	HKPluginGraphicsVulkanImpl* plugin = new HKPluginGraphicsVulkanImpl();
	plugin->addRef();
	return plugin;
}