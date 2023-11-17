#ifndef HK_SHAPE_MESH__H
#define HK_SHAPE_MESH__H
#include "../shape.h"
#include "../value_array.h"
#include "../object_array.h"

typedef enum HKMeshTopology {
	HKMeshTopologyTriangles,
	HKMeshTopologyQuads,
	HKMeshTopologyLines,
	HKMeshTopologyLineStrip,
	HKMeshTopologyPoints
} HKMeshTopology;

// {94DFE5FC-1886-4506-AB58-91AC8AFC68E2}
#define HK_OBJECT_TYPEID_Mesh         HK_UUID_DEFINE(0x94dfe5fc, 0x1886, 0x4506, 0xab, 0x58, 0x91, 0xac, 0x8a, 0xfc, 0x68, 0xe2)
#define HK_OBJECT_TYPEID_SubMesh      HK_UUID_DEFINE(0x83fbfb86, 0x2150, 0x4dd2, 0x81, 0xd6, 0x2f, 0x62, 0xe2, 0x1f,  0x0, 0xa8)
#define HK_OBJECT_TYPEID_ArrayMesh    HK_UUID_DEFINE(0x9f715414, 0xa630, 0x44cc, 0x86, 0xb7, 0x4f, 0x52, 0xe5, 0xdf, 0x67, 0x2f)
#define HK_OBJECT_TYPEID_ArraySubMesh HK_UUID_DEFINE(0x3aa4c7e4, 0x70b0, 0x4daa, 0xb2, 0xbd, 0xbb, 0xec, 0x4 , 0x3e, 0x7 , 0xbc)

#if defined(__cplusplus)
struct HKArrayMesh;
struct HKArraySubMesh;
struct HKMesh;

struct HKSubMesh : public HKShape {
	virtual HKMeshTopology HK_API getTopologoy() const                 = 0;
	virtual HKMesh*        HK_API internal_getMesh()                   = 0; 
	virtual const HKMesh*  HK_API internal_getMesh_const() const       = 0; 
	HK_INLINE HKMesh*             getMesh()       { return internal_getMesh(); }
	HK_INLINE const HKMesh*       getMesh_const() { return internal_getMesh_const(); }
};
struct HKMesh    : public HKShape {
	static HK_INLINE HKMesh* create();
	virtual HKU32           HK_API getSubMeshCount() const = 0;
	virtual HKArraySubMesh* HK_API getSubMeshes()          = 0;
	virtual HKMeshTopology  HK_API getTopologoy(HKU32 submesh_idx) const = 0;
	virtual HKU32           HK_API getVertexCount() const  = 0;
	virtual void            HK_API setVertexCount(HKU32 )  = 0;
	virtual HKArrayVec3*    HK_API getVertices()const = 0;
	virtual void            HK_API setVertices(const HKArrayVec3* ) = 0;
	virtual HKArrayVec3*    HK_API getNormals()const = 0;
	virtual void            HK_API setNormals(const HKArrayVec3*) = 0;
	virtual HKBool          HK_API hasNormal ()const = 0;
};

#else
typedef struct HKMesh    HKMesh;
typedef struct HKSubMesh HKSubMesh;
#endif

HK_NAMESPACE_TYPE_ALIAS(Mesh);
HK_NAMESPACE_TYPE_ALIAS(SubMesh);

HK_SHAPE_ARRAY_DEFINE(Mesh);
HK_SHAPE_ARRAY_DEFINE(SubMesh);

HK_EXTERN_C HK_DLL HKMesh*          HK_API HKSubMesh_internal_getMesh(HKSubMesh* mesh);
HK_EXTERN_C HK_DLL const HKMesh*    HK_API HKSubMesh_internal_getMesh_const(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL HKMeshTopology   HK_API HKSubMesh_getTopology(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL HKMesh*          HK_API HKMesh_create(void);
HK_EXTERN_C HK_DLL HKU32            HK_API HKMesh_getSubMeshCount(const HKMesh* mesh);
HK_EXTERN_C HK_DLL HKArraySubMesh*  HK_API HKMesh_getSubMeshes(HKMesh* mesh);
HK_EXTERN_C HK_DLL HKMeshTopology   HK_API HKMesh_getTopologoy(const HKMesh* mesh, HKU32 submesh_idx);
HK_EXTERN_C HK_DLL HKU32            HK_API HKMesh_getVertexCount(const HKMesh* mesh) ;
HK_EXTERN_C HK_DLL void             HK_API HKMesh_setVertexCount(HKMesh* mesh,HKU32) ;
HK_EXTERN_C HK_DLL HKArrayVec3*     HK_API HKMesh_getVertices (const HKMesh* mesh);
HK_EXTERN_C HK_DLL void             HK_API HKMesh_setVertices(HKMesh* mesh, const HKArrayVec3*);
HK_EXTERN_C HK_DLL HKArrayVec3*     HK_API HKMesh_getNormals (const HKMesh* mesh);
HK_EXTERN_C HK_DLL void             HK_API HKMesh_setNormals(HKMesh* mesh, const HKArrayVec3*);
HK_EXTERN_C HK_DLL HKBool           HK_API HKMesh_hasNormal(const HKMesh* mesh);
#if defined(__cplusplus)
HK_INLINE HKMesh* HKMesh::create() { return HKMesh_create(); }
#endif

#endif