#ifndef HK_SHAPE_OBJ_MESH__H
#define HK_SHAPE_OBJ_MESH__H
#include "mesh.h"
// {4817E141-01F5-4D49-AB31-B535BC91BFE5}
#define HK_OBJECT_TYPEID_ObjMesh         HK_UUID_DEFINE(0x4817e141, 0x1f5 , 0x4d49, 0xab, 0x31, 0xb5, 0x35, 0xbc, 0x91, 0xbf, 0xe5)
#define HK_OBJECT_TYPEID_ObjSubMesh      HK_UUID_DEFINE(0x286f02e2, 0x2a50, 0x4db3, 0x9d, 0x58, 0xea, 0xe5, 0xd1, 0xb1, 0x42, 0x24)
#define HK_OBJECT_TYPEID_ArrayObjMesh    HK_UUID_DEFINE(0x4c653789, 0xb28e, 0x42d1, 0x9e, 0x99, 0x3a, 0xe2, 0x8a, 0xa6, 0xeb, 0x72)
#define HK_OBJECT_TYPEID_ArrayObjSubMesh HK_UUID_DEFINE(0xf29055f9, 0x2c72, 0x4d5a, 0x83, 0x1b, 0x9e, 0xa5, 0x2f, 0xf , 0xe3, 0x42)

#if defined(__cplusplus)
struct HKObjMesh;
struct HKObjSubMesh : public HKSubMesh {
	virtual HKCStr HK_API getName() const = 0;
};
struct HKObjMesh    : public HKMesh    {
	static HK_INLINE HKObjMesh* create() ;
	virtual void   HK_API setFilename(HKCStr filename)            = 0;
	virtual HKCStr HK_API getFilename() const                     = 0;
	virtual HKBool HK_API loadFile(HKCStr filename)               = 0;
	virtual HKCStr HK_API getSubMeshName(HKU32 submesh_idx) const = 0;
};
#else
typedef struct HKObjSubMesh HKObjSubMesh;
typedef struct HKObjMesh    HKObjMesh;
#endif

HK_SHAPE_ARRAY_DEFINE(ObjMesh);
HK_SHAPE_ARRAY_DEFINE(ObjSubMesh);

HK_EXTERN_C HK_DLL HKCStr     HK_API HKObjSubMesh_getName(const HKObjSubMesh* obj_mesh);
HK_OBJSUBMESH_C_DERIVE_METHODS(HKObjSubMesh);

HK_EXTERN_C HK_DLL HKObjMesh* HK_API HKObjMesh_create();
HK_EXTERN_C HK_DLL HKCStr     HK_API HKObjMesh_getFilename(const HKObjMesh* obj_mesh);
HK_EXTERN_C HK_DLL void       HK_API HKObjMesh_setFilename(      HKObjMesh* obj_mesh,HKCStr filename);
HK_EXTERN_C HK_DLL HKBool     HK_API HKObjMesh_loadFile(HKObjMesh* obj_mesh, HKCStr filename);
HK_EXTERN_C HK_DLL HKCStr     HK_API HKObjMesh_getSubMeshName(const HKObjMesh* obj_mesh, HKU32 submesh_idx) ;
HK_OBJMESH_C_DERIVE_METHODS(HKObjMesh);

#if defined(__cplusplus)
HK_INLINE HKObjMesh* HKObjMesh::create() { return HKObjMesh_create(); }
#endif

HK_NAMESPACE_TYPE_ALIAS(ObjMesh);
HK_NAMESPACE_TYPE_ALIAS(ObjSubMesh);


#endif
