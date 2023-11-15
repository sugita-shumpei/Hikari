#define HK_DLL_EXPORT
#include "shape.h"

HK_EXTERN_C HK_DLL HKCAabb HK_API HKShape_getAabb(const HKShape* shape)
{
	if (shape) {
		return shape->getAabb();
	}
	else {
		return HKAabb();
	}
}
