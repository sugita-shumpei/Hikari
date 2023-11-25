#define HK_DLL_EXPORT
#include <hikari/graphics/instance.h>


HK_EXTERN_C HK_DLL HKCStr HKGraphicsInstance_getApiName(const HKGraphicsInstance* gi) {
	if (gi) { return gi->getApiName(); }
	else { return ""; }
}