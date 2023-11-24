#define HK_DLL_EXPORT
#include <hikari/shape.h>
#include <hikari/ref_cnt_object.h>
#include <vector>

HK_EXTERN_C HK_DLL HKCAabb HK_API HKShape_getAabb(const HKShape *shape)
{
	if (shape)
	{
		return shape->getAabb();
	}
	else
	{
		return HKAabb();
	}
}


HK_SHAPE_ARRAY_IMPL_DEFINE(Shape);