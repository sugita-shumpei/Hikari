#define  HK_DLL_EXPORT
#include <hikari/plugin.h>
#include <hikari/shape/plugin.h>

#include <hikari/ref_cnt_object.h>
#include <hikari/shape/sphere.h>
#include <hikari/shape/mesh.h>
#include <hikari/shape/obj_mesh.h>

struct HK_DLL HKPluginShapeImpl : public HKPlugin, protected HKRefCntObject {
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
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin || iid == HK_OBJECT_TYPEID_PluginShape)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown* HK_API createObject(HKUUID iid)  override
	{
		if (iid == HK_OBJECT_TYPEID_Sphere)     { return HKSphere_create();  }
		if (iid == HK_OBJECT_TYPEID_Mesh)       { return HKMesh_create();    }
		if (iid == HK_OBJECT_TYPEID_ObjMesh)    { return HKObjMesh_create(); }
		if (iid == HK_OBJECT_TYPEID_ArraySphere ) { return HKArraySphere_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMesh   ) { return HKArrayMesh_create()   ; }
		if (iid == HK_OBJECT_TYPEID_ArraySubMesh) { return HKArraySubMesh_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayObjMesh   ) { return HKArrayObjMesh_create()   ; }
		if (iid == HK_OBJECT_TYPEID_ArrayObjSubMesh) { return HKArrayObjSubMesh_create(); }

		return nullptr;
	}
	void        HK_API destroyObject()     override
	{
		return;
	}
	// HKPlugin ‚ð‰î‚µ‚ÄŒp³‚³‚ê‚Ü‚µ‚½
	HKUUID      HK_API getID() const override
	{
		return HK_OBJECT_TYPEID_PluginShape;
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
	if (manager->contain(HK_OBJECT_TYPEID_PluginShape)) { return nullptr; }
	HKPluginShapeImpl* plugin = new HKPluginShapeImpl();
	plugin->addRef();
	return plugin;
}