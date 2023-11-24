#ifndef HK_SHAPE_MESH__H
#define HK_SHAPE_MESH__H
#if !defined(__CUDACC__)

#include <hikari/shape/mesh_utils.h>
#include <hikari/shape.h>
#include <hikari/value_array.h>
#include <hikari/object_array.h>

// {94DFE5FC-1886-4506-AB58-91AC8AFC68E2}
#define HK_OBJECT_TYPEID_Mesh         HK_UUID_DEFINE(0x94dfe5fc, 0x1886, 0x4506, 0xab, 0x58, 0x91, 0xac, 0x8a, 0xfc, 0x68, 0xe2)
#define HK_OBJECT_TYPEID_SubMesh      HK_UUID_DEFINE(0x83fbfb86, 0x2150, 0x4dd2, 0x81, 0xd6, 0x2f, 0x62, 0xe2, 0x1f,  0x0, 0xa8)
#define HK_OBJECT_TYPEID_ArrayMesh    HK_UUID_DEFINE(0x9f715414, 0xa630, 0x44cc, 0x86, 0xb7, 0x4f, 0x52, 0xe5, 0xdf, 0x67, 0x2f)
#define HK_OBJECT_TYPEID_ArraySubMesh HK_UUID_DEFINE(0x3aa4c7e4, 0x70b0, 0x4daa, 0xb2, 0xbd, 0xbb, 0xec, 0x4 , 0x3e, 0x7 , 0xbc)

typedef enum HKMeshTopology {
	HKMeshTopologyTriangles,
	HKMeshTopologyQuads,
	HKMeshTopologyLines,
	HKMeshTopologyLineStrips,
	HKMeshTopologyPoints
} HKMeshTopology;

HK_INLINE HK_CXX11_CONSTEXPR HKU32 HKMeshTopology_getVertexCount(HKMeshTopology topology) {
	switch (topology)
	{
	case HKMeshTopologyTriangles: return 3;
		break;
	case HKMeshTopologyQuads:     return 4;
		break;
	case HKMeshTopologyLines:     return 2;
		break;
	case HKMeshTopologyLineStrips: return 1;
		break;
	case HKMeshTopologyPoints:    return 1;
		break;
	default: return 0;
		break;
	}
}

#if defined(__cplusplus)
struct HKArrayMesh;
struct HKArraySubMesh;
struct HKMesh;
struct HKSubMesh : public HKShape {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_SubMesh; }
	virtual void           HK_API clear() = 0;
	virtual void           HK_API setIndices(const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds) = 0;
	virtual HKArrayU32*    HK_API getIndices(HKBool add_base_vertex) const = 0;
	virtual HKU32          HK_API getVertexCount() const               = 0;
	virtual HKU32          HK_API getIndexCount() const                = 0;
	virtual HKMeshTopology HK_API getTopology() const                 = 0;
	virtual HKU32          HK_API getBaseVertex() const                = 0;
	virtual HKU32          HK_API getFirstVertex() const               = 0;
	virtual HKMesh*        HK_API internal_getMesh()                   = 0; 
	virtual const HKMesh*  HK_API internal_getMesh_const() const       = 0;
	virtual void           HK_API updateAabb()                         = 0;

	HK_INLINE HKArrayU32*         getIndices() const { return getIndices(true); }
	HK_INLINE void                setIndices(const HKArrayU32* indices) {  setIndices(indices,HKMeshTopologyTriangles,0, true); }
	HK_INLINE void                setIndices(const HKArrayU32* indices, HKU32 base_vertex) { setIndices(indices, HKMeshTopologyTriangles, base_vertex, true); }
	HK_INLINE void                setIndices(const HKArrayU32* indices, HKMeshTopology topology) { setIndices(indices, topology, 0, true); }
	HK_INLINE void                setIndices(const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex) { setIndices(indices, topology, 0,true); }

	HK_INLINE HKMesh*             getMesh()       { return internal_getMesh(); }
	HK_INLINE const HKMesh*       getMesh() const { return internal_getMesh_const(); }
};
struct HKMesh    : public HKShape {
	static HK_CXX11_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_Mesh; }
	virtual void            HK_API clear() = 0;
	virtual HKMesh*         HK_API clone() const = 0;
	virtual void            HK_API copy (const HKMesh* mesh) = 0;
	virtual HKU32           HK_API getSubMeshCount() const = 0;
	virtual void            HK_API setSubMeshCount(HKU32 ) = 0;
	virtual HKArraySubMesh* HK_API getSubMeshes()          = 0;
	virtual HKArrayU32*     HK_API getSubMeshIndices(HKU32 submesh_idx, HKBool add_base_vertex)  const = 0;
	virtual void            HK_API setSubMeshIndices(HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds) = 0;
	virtual HKMeshTopology  HK_API getSubMeshTopology(  HKU32 submesh_idx) const = 0;
	virtual HKU32           HK_API getSubMeshIndexCount(HKU32 submesh_idx) const = 0;
	virtual HKU32           HK_API getSubMeshBaseVertex(HKU32 submesh_idx) const = 0;
	virtual HKU32           HK_API getVertexCount() const  = 0;
	virtual void            HK_API setVertexCount(HKU32 )  = 0;
	virtual HKArrayVec3*    HK_API getVertices()const = 0;
	virtual void            HK_API setVertices(const HKArrayVec3*) = 0;
	virtual HKVec3          HK_API getVertex(HKU32)const = 0;
	virtual HKArrayVec3*    HK_API getNormals()const = 0;
	virtual void            HK_API setNormals(const HKArrayVec3*) = 0;
	virtual HKArrayVec4*    HK_API getTangents()const = 0;
	virtual void            HK_API setTangents(const HKArrayVec4*) = 0;
	virtual HKArrayColor*   HK_API getColors()const = 0;
	virtual void            HK_API setColors(const HKArrayColor*) = 0;
	virtual HKArrayColor8*  HK_API getColor8s()const = 0;
	virtual void            HK_API setColor8s(const HKArrayColor8*) = 0;
	virtual HKBool          HK_API hasNormal()const = 0;
	virtual HKBool          HK_API hasTangent()const = 0;
	virtual HKBool          HK_API hasColor()const = 0;
	virtual HKBool          HK_API hasUV( HKU32 idx)const = 0;
	virtual HKArrayVec2*    HK_API getUVs(HKU32 idx)const = 0;
	virtual void            HK_API setUVs(HKU32 idx,const HKArrayVec2*) = 0;
	virtual HKArrayU32*     HK_API getIndices(HKBool add_base_vertex) const = 0;
	virtual void            HK_API updateAabb() = 0;

	HK_INLINE HKArrayVec2* getUVs ()const { return getUVs(0); }
	HK_INLINE HKArrayVec2* getUV0s()const { return getUVs(0); }
	HK_INLINE HKArrayVec2* getUV1s()const { return getUVs(1); }
	HK_INLINE HKArrayVec2* getUV2s()const { return getUVs(2); }
	HK_INLINE HKArrayVec2* getUV3s()const { return getUVs(3); }
	HK_INLINE HKArrayVec2* getUV4s()const { return getUVs(4); }
	HK_INLINE HKArrayVec2* getUV5s()const { return getUVs(5); }
	HK_INLINE HKArrayVec2* getUV6s()const { return getUVs(6); }
	HK_INLINE HKArrayVec2* getUV7s()const { return getUVs(7); }

	HK_INLINE void         setUVs (const HKArrayVec2* uv) { setUVs(0, uv); }
	HK_INLINE void         setUV0s(const HKArrayVec2* uv) { setUVs(0, uv); }
	HK_INLINE void         setUV1s(const HKArrayVec2* uv) { setUVs(1, uv); }
	HK_INLINE void         setUV2s(const HKArrayVec2* uv) { setUVs(2, uv); }
	HK_INLINE void         setUV3s(const HKArrayVec2* uv) { setUVs(3, uv); }
	HK_INLINE void         setUV4s(const HKArrayVec2* uv) { setUVs(4, uv); }
	HK_INLINE void         setUV5s(const HKArrayVec2* uv) { setUVs(5, uv); }
	HK_INLINE void         setUV6s(const HKArrayVec2* uv) { setUVs(6, uv); }
	HK_INLINE void         setUV7s(const HKArrayVec2* uv) { setUVs(7, uv); }
	
	HK_INLINE HKArrayU32*  getIndices()  const { return getIndices(true); }
	HK_INLINE HKArrayU32*  getIndices(HKU32 submesh_idx)  const { return getSubMeshIndices(submesh_idx, true); }
	HK_INLINE HKArrayU32*  getIndices(HKU32 submesh_idx, HKBool add_base_vertex)  const { return getSubMeshIndices(submesh_idx, add_base_vertex); }
	HK_INLINE void         setIndices(const HKArrayU32* indices) { return setSubMeshIndices(0, indices, HKMeshTopologyTriangles, 0,true); }
	HK_INLINE void         setIndices(HKU32 submesh_idx, const HKArrayU32* indices) { return setSubMeshIndices(submesh_idx, indices, HKMeshTopologyTriangles, 0, true); }
	HK_INLINE void         setIndices(HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology) { return setSubMeshIndices(submesh_idx, indices, topology, 0, true); }
	HK_INLINE void         setIndices(HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex) { return setSubMeshIndices(submesh_idx, indices, topology, base_vertex, true); }
};
#else
typedef struct HKMesh    HKMesh;
typedef struct HKSubMesh HKSubMesh;
#endif

HK_NAMESPACE_TYPE_ALIAS(Mesh);
HK_NAMESPACE_TYPE_ALIAS(SubMesh);

HK_SHAPE_ARRAY_DEFINE(Mesh);
HK_SHAPE_ARRAY_DEFINE(SubMesh);

HK_EXTERN_C HK_DLL_FUNCTION HKMesh*          HK_DLL_FUNCTION_NAME(HKSubMesh_internal_getMesh)(HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION const HKMesh*    HK_DLL_FUNCTION_NAME(HKSubMesh_internal_getMesh_const)(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKSubMesh_clear)(HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKSubMesh_setIndices)(HKSubMesh* mesh, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayU32*      HK_DLL_FUNCTION_NAME(HKSubMesh_getIndices)(const HKSubMesh* mesh, HKBool add_base_vertex);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKSubMesh_getVertexCount)(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKSubMesh_getIndexCount)(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKMeshTopology   HK_DLL_FUNCTION_NAME(HKSubMesh_getTopology)(const HKSubMesh* mesh) ;
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKSubMesh_getBaseVertex)(const HKSubMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKSubMesh_getFirstVertex)(const HKSubMesh* mesh);
HK_SHAPE_C_DERIVE_METHODS(HKSubMesh);

HK_EXTERN_C HK_DLL_FUNCTION HKMesh*          HK_DLL_FUNCTION_NAME(HKMesh_create)(void);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_clear)(HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshCount)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setSubMeshCount)(HKMesh* mesh, HKU32 count);
HK_EXTERN_C HK_DLL_FUNCTION HKArraySubMesh*  HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshes)(HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKMeshTopology   HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshTopology)( const HKMesh* mesh, HKU32 submesh_idx);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshIndexCount)(const HKMesh* mesh, HKU32 submesh_idx);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshBaseVertex)(const HKMesh* mesh, HKU32 submesh_idx);
HK_EXTERN_C HK_DLL_FUNCTION HKU32            HK_DLL_FUNCTION_NAME(HKMesh_getVertexCount)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setVertexCount)(HKMesh* mesh,HKU32);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayVec3*     HK_DLL_FUNCTION_NAME(HKMesh_getVertices )(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setVertices)(HKMesh* mesh, const HKArrayVec3*);
HK_EXTERN_C HK_DLL_FUNCTION HKCVec3          HK_DLL_FUNCTION_NAME(HKMesh_getVertex)(const HKMesh* mesh, HKU32 idx);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayVec3*     HK_DLL_FUNCTION_NAME(HKMesh_getNormals )(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setNormals)(HKMesh* mesh, const HKArrayVec3*);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKMesh_hasNormal)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayVec4*     HK_DLL_FUNCTION_NAME(HKMesh_getTangents)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setTangents)(HKMesh* mesh, const HKArrayVec4*);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKMesh_hasTangent)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayColor*    HK_DLL_FUNCTION_NAME(HKMesh_getColors)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setColors)(HKMesh* mesh, const HKArrayColor*);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayColor8*   HK_DLL_FUNCTION_NAME(HKMesh_getColor8s)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setColor8s)(HKMesh* mesh, const HKArrayColor8*);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKMesh_hasColor)(const HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKBool           HK_DLL_FUNCTION_NAME(HKMesh_hasUV)(const HKMesh* mesh,HKU32 idx);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayVec2*     HK_DLL_FUNCTION_NAME(HKMesh_getUVs)(const HKMesh* mesh,HKU32 idx);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setUVs)(HKMesh* mesh, HKU32 idx,const HKArrayVec2*);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayU32*      HK_DLL_FUNCTION_NAME(HKMesh_getIndices)(const HKMesh* mesh, HKBool add_base_vertex);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_updateAabb)(HKMesh* mesh);
HK_EXTERN_C HK_DLL_FUNCTION HKArrayU32*      HK_DLL_FUNCTION_NAME(HKMesh_getSubMeshIndices)(const HKMesh* mesh, HKU32 submesh_idx, HKBool add_base_vertex);
HK_EXTERN_C HK_DLL_FUNCTION void             HK_DLL_FUNCTION_NAME(HKMesh_setSubMeshIndices)(HKMesh* mesh, HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds);
HK_SHAPE_C_DERIVE_METHODS(HKMesh);

#if defined(__cplusplus)
HK_OBJECT_CREATE_TRAITS(HKMesh);
#endif

#endif
#endif