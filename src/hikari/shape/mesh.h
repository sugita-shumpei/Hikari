#ifndef HK_SHAPE_MESH__H
#define HK_SHAPE_MESH__H
#include "../shape.h"

struct HKMesh  : public HKShape {
	virtual       HKVec3* internal_getPosition()             = 0;
	virtual const HKVec3* internal_getPosition_const()       = 0;
	virtual       HKVec3* internal_getNormal()               = 0;
	virtual const HKVec3* internal_getNormal_const()         = 0;
	virtual       HKVec3* internal_getTangent()              = 0;
	virtual const HKVec3* internal_getTangent_const()        = 0;
	virtual       HKVec3* internal_getUV(HKU32 idx)          = 0;
	virtual const HKVec3* internal_getUV_const(HKU32 idx)    = 0;
	virtual       HKVec3* internal_getColor(HKU32 idx)       = 0;
	virtual const HKVec3* internal_getColor_const(HKU32 idx) = 0;

};
// Read Write Mesh
struct HKRwMesh {

};

#endif