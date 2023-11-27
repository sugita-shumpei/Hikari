#define HK_DLL_EXPORT
#include <hikari/graphics/opengl/plugin.h>
#include <memory>
#include <hikari/ref_cnt_object.h>
#include <hikari/graphics/plugin.h>
#include <hikari/graphics/opengl/entry.h>
#include <hikari/graphics/opengl/entry_impl.h>
#include <hikari/graphics/opengl/context.h>
struct HK_DLL HKPluginGraphicsOpenGLImpl : public HKPlugin, protected HKRefCntObject {
	HKPluginGraphicsOpenGLImpl() noexcept : HKPlugin{}, HKRefCntObject{} {}
	virtual ~HKPluginGraphicsOpenGLImpl() {}

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
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin || iid == HK_OBJECT_TYPEID_PluginGraphicsOpenGL)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown* HK_API createObject(HKUUID iid)  override
	{
		//if (iid == HK_OBJECT_TYPEID_GraphicsOpenGLContextManager) {
		//	return HKGraphicsOpenGLContextManager_create();
		//}
		return nullptr;
	}
	void        HK_API destroyObject()     override
	{
		return;
	}
	// HKPlugin ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
	HKUUID      HK_API getID() const override
	{
		return HK_OBJECT_TYPEID_PluginGraphicsOpenGL;
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
	HKPluginGraphicsOpenGLImpl* plugin = new HKPluginGraphicsOpenGLImpl();
	plugin->addRef();
	return plugin;
}