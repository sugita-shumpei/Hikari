#define  HK_DLL_EXPORT
#include <hikari/plugin.h>
#include <hikari/ref_cnt_object.h>
#include <hikari/object_array.h>
#include <hikari/value_array.h>
#include <hikari/shape.h>

struct HK_DLL HKPluginImpl_hikari : public HKPlugin, protected HKRefCntObject {
	HKU32      HK_API addRef() override
	{
		return HKRefCntObject::addRef();
	}
	HKU32      HK_API release() override
	{
		return HKRefCntObject::release();
	}
	HKBool     HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Plugin)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	HKUnknown* HK_API createObject(HKUUID iid) const override
	{
		if (iid == HK_OBJECT_TYPEID_ArrayShape) { return HKArrayShape_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayUnknown) { return HKArrayUnknown_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayByte) { return HKArrayByte_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayChar) { return HKArrayChar_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU8 )  { return HKArrayU8_create() ; }
		if (iid == HK_OBJECT_TYPEID_ArrayU16)  { return HKArrayU16_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU32)  { return HKArrayU32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayU64)  { return HKArrayU64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI8 )  { return HKArrayI8_create() ; }
		if (iid == HK_OBJECT_TYPEID_ArrayI16)  { return HKArrayI16_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI32)  { return HKArrayI32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayI64)  { return HKArrayI64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayF32)  { return HKArrayF32_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayF64)  { return HKArrayF64_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec2) { return HKArrayVec2_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec3) { return HKArrayVec3_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayVec4) { return HKArrayVec4_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat2x2) { return HKArrayMat2x2_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat3x3) { return HKArrayMat3x3_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayMat4x4) { return HKArrayMat4x4_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayColor ) { return HKArrayColor_create(); }
		if (iid == HK_OBJECT_TYPEID_ArrayColor8) { return HKArrayColor8_create(); }

		return nullptr;
	}

	void HK_API destroyObject() override
	{
		return;
	}
};

HK_EXTERN_C HK_DLL HKPlugin* HK_API HKPlugin_create() {
	HKPluginImpl_hikari* res = new HKPluginImpl_hikari();
	res->addRef();
	return res;
}