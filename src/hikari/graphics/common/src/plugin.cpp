#define  HK_DLL_EXPORT
#include <hikari/graphics/plugin.h>
#include <hikari/graphics/instance.h>
#include <hikari/ref_cnt_object.h>

struct HK_DLL HKPluginGraphicsCommonImpl : public HKPlugin, protected HKRefCntObject {
	HKPluginGraphicsCommonImpl() noexcept : HKPlugin{}, HKRefCntObject{} {}
	virtual ~HKPluginGraphicsCommonImpl() {}

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
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin || iid == HK_OBJECT_TYPEID_PluginGraphicsCommon)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown*  HK_API createObject(HKUUID iid)  override
	{
		return nullptr;
	}
	void        HK_API destroyObject()     override
	{
		return;
	}
	// HKPlugin ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
	HKUUID      HK_API getID() const override
	{
		return HK_OBJECT_TYPEID_PluginGraphicsCommon;
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

HK_EXTERN_C HK_DLL HKPlugin* HK_API HKPlugin_create(HKPluginManager* manager) {
	if (!manager) { return nullptr; }
	if (manager->contain(HK_OBJECT_TYPEID_Unknown)) { return nullptr; }
	HKPluginGraphicsCommonImpl* plugin = new HKPluginGraphicsCommonImpl();
	plugin->addRef();
	return plugin;
}