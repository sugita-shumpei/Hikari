#ifndef HK_GRAPHICS_OPENGL_CONTEXT__H
#define HK_GRAPHICS_OPENGL_CONTEXT__H
#if !defined(__CUDACC__)
#include <hikari/object.h>
#define HK_OBJECT_TYPEID_GraphicsOpenGLContext        HK_UUID_DEFINE(0x3d8dfbd4, 0xb7ad, 0x4b8b, 0xad, 0x37, 0x3e, 0x5b, 0x70, 0x64, 0x45, 0x81)
#define HK_OBJECT_TYPEID_GraphicsOpenGLContextManager HK_UUID_DEFINE(0x20c6eb2a, 0xd139, 0x4ef1, 0x9c, 0xef, 0xd5, 0xf2, 0x7a, 0x86, 0x42, 0x1b)
#if defined(__cplusplus)
struct HKGraphicsOpenGLContext : public HKUnknown {
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsOpenGLContext; }
	virtual HKU64 HK_API getThreadID() const    = 0;
	virtual void  HK_API internal_setCurrent() = 0;
	virtual void  HK_API internal_popCurrent() = 0;
};
struct HKGraphicsOpenGLContextManager : public HKUnknown {
	static HK_INLINE HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_GraphicsOpenGLContextManager; }
	virtual HKU64                    HK_API getThreadID() const                          = 0;
	virtual HKGraphicsOpenGLContext* HK_API setContext(HKGraphicsOpenGLContext* context) = 0;
	virtual HKGraphicsOpenGLContext* HK_API getContext()                                 = 0;
};
#else
typedef struct HKGraphicsOpenGLContext        HKGraphicsOpenGLContext;
typedef struct HKGraphicsOpenGLContextManager HKGraphicsOpenGLContextManager;
#endif

HK_EXTERN_C HK_DLL_FUNCTION HKU64                           HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContext_getThreadID)(const HKGraphicsOpenGLContext*);
HK_EXTERN_C HK_DLL_FUNCTION void                            HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContext_internal_setCurrent)(HKGraphicsOpenGLContext*);
HK_EXTERN_C HK_DLL_FUNCTION void                            HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContext_internal_popCurrent)(HKGraphicsOpenGLContext*);
HK_EXTERN_C HK_DLL_FUNCTION HKGraphicsOpenGLContextManager* HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContextManager_create)();
HK_EXTERN_C HK_DLL_FUNCTION HKU64                           HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContextManager_getThreadID)(const HKGraphicsOpenGLContextManager*);
HK_EXTERN_C HK_DLL_FUNCTION HKGraphicsOpenGLContext*        HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContextManager_setContext )(HKGraphicsOpenGLContextManager*, HKGraphicsOpenGLContext*);
HK_EXTERN_C HK_DLL_FUNCTION HKGraphicsOpenGLContext*        HK_DLL_FUNCTION_NAME(HKGraphicsOpenGLContextManager_getContext )(HKGraphicsOpenGLContextManager*);

#endif
#endif
