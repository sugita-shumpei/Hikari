#define HK_DLL_EXPORT 
#include <hikari/graphics/opengl/context.h>
#include <utility>
#include <thread>
#include <hikari/ref_cnt_object.h>

struct HK_DLL HKGraphicsOpenGLContextManagerImpl : public HKGraphicsOpenGLContextManager, protected HKRefCntObject {
	HKGraphicsOpenGLContextManagerImpl() :HKGraphicsOpenGLContextManager(), HKRefCntObject(), m_thread_id{}, m_context{ nullptr } {
		auto hsh = std::hash<std::thread::id>();
		m_thread_id = hsh (std::this_thread::get_id());
	}
	virtual HK_API ~HKGraphicsOpenGLContextManagerImpl(){}
	
	virtual HKU32  HK_API addRef() override
	{
		return HKRefCntObject::addRef();
	}
	virtual HKU32  HK_API release() override
	{
		return HKRefCntObject::release();
	}
	virtual HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_GraphicsOpenGLContextManager) {
			HKRefCntObject::addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	virtual HKU64                    HK_API getThreadID() const override { return m_thread_id; }
	virtual HKGraphicsOpenGLContext* HK_API setContext(HKGraphicsOpenGLContext* context) override { 
		if (!context) {
			HKGraphicsOpenGLContext* old = m_context;
			if (m_context) {
				m_context->internal_popCurrent();
				m_context = nullptr;
			}
			return old;
		}
		else {
			if (m_thread_id == context->getThreadID()) {
				if (m_context == context) { return m_context; }
				if (m_context) {
					m_context->internal_popCurrent();
				}
				context->internal_setCurrent();
				HKGraphicsOpenGLContext* old = m_context;
				m_context = context;
				return old;
			}
			else {
				return m_context;
			}
		}
	}
	virtual HKGraphicsOpenGLContext* HK_API getContext() override { return nullptr; }
	virtual void HK_API destroyObject() override
	{
		if (m_context) { m_context->internal_popCurrent();  m_context = nullptr; }
		return;
	}
	HKU64                    m_thread_id;
	HKGraphicsOpenGLContext* m_context  ;
};
HK_EXTERN_C HK_DLL HKU64                           HK_API HKGraphicsOpenGLContext_getThreadID(const HKGraphicsOpenGLContext* context) {
	if (context) {
		return context->getThreadID();
	}
	else {
		return 0;
	}
}
HK_EXTERN_C HK_DLL void                            HK_API HKGraphicsOpenGLContext_internal_setCurrent(HKGraphicsOpenGLContext* context) {
	if (context) {
		return context->internal_setCurrent();
	}
}
HK_EXTERN_C HK_DLL void                            HK_API HKGraphicsOpenGLContext_internal_popCurrent(HKGraphicsOpenGLContext* context) {
	if (context) {
		return context->internal_popCurrent();
	}
}
HK_EXTERN_C HK_DLL HKGraphicsOpenGLContextManager* HK_API HKGraphicsOpenGLContextManager_create() {
	auto manager = new HKGraphicsOpenGLContextManagerImpl();
	manager->addRef();
	return manager;
}
HK_EXTERN_C HK_DLL HKU64                           HK_API HKGraphicsOpenGLContextManager_getThreadID(const HKGraphicsOpenGLContextManager* manager) {
	if (manager) {
		return manager->getThreadID();
	}
	else {
		return 0;
	}
}
HK_EXTERN_C HK_DLL HKGraphicsOpenGLContext*        HK_API HKGraphicsOpenGLContextManager_setContext(HKGraphicsOpenGLContextManager* manager, HKGraphicsOpenGLContext* context) {
	if (manager) {
		return manager->setContext(context);
	}
	else {
		return nullptr;
	}
}
HK_EXTERN_C HK_DLL HKGraphicsOpenGLContext*        HK_API HKGraphicsOpenGLContextManager_getContext(HKGraphicsOpenGLContextManager* manager) {
	if (manager) {
		return manager->getContext();
	}
	else {
		return nullptr;
	}
}