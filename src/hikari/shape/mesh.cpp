#define HK_DLL_EXPORT
#include "mesh.h"
#include "ref_cnt_object.h"
#include <vector>

struct HK_DLL HKSubMeshImpl : public HKSubMesh, protected HKRefCntObject {
	HKSubMeshImpl(HKMesh* base) noexcept :HKSubMesh(), HKRefCntObject(), mesh{ base }, topology{HKMeshTopologyTriangles} {}
	virtual HK_API ~HKSubMeshImpl() {}

	static auto create(HKMesh* base) -> HKSubMeshImpl* { auto res = new HKSubMeshImpl(base); res->addRef(); return res; }
	virtual HKU32          HK_API addRef() override  { return HKRefCntObject::addRef(); }
	virtual HKU32          HK_API release() override { return HKRefCntObject::release(); }
	virtual HKBool         HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_SubMesh) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	virtual HKMesh*        HK_API internal_getMesh()       override { return mesh; }
	virtual const HKMesh*  HK_API internal_getMesh_const() const override { return mesh; }
	virtual HKMeshTopology HK_API getTopologoy() const override
	{
		return topology;
	}
	virtual void           HK_API destroyObject() override { }
	virtual HKAabb         HK_API getAabb() const override { return HKAabb(); }
	HKMesh* mesh;
	HKMeshTopology topology;
};
struct HKMeshImpl : public HKMesh, protected HKRefCntObject {
	HKMeshImpl() noexcept : HKMesh(), HKRefCntObject(), m_submeshes{ HKSubMeshImpl::create(this)}, m_vertex_count{ 0 }, m_has_normal{ false } {}
	virtual HK_API ~HKMeshImpl() {}
	virtual HKU32           HK_API addRef() override  { return HKRefCntObject::addRef(); }
	virtual HKU32           HK_API release() override { return HKRefCntObject::release(); }
	virtual HKBool          HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_Mesh) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	virtual HKU32           HK_API getSubMeshCount() const override { return m_submeshes.size(); }
	virtual HKArraySubMesh* HK_API getSubMeshes() override {
		auto res = HKArraySubMesh::create();
		res->resize(m_submeshes.size());
		HKU32 i = 0;
		for (auto& submesh : m_submeshes) {
			res->setValue(i, submesh);
			++i;
		}
		return res;
	}
	virtual HKU32           HK_API getVertexCount() const override
	{
		return m_vertex_count;
	}
	virtual void            HK_API setVertexCount(HKU32 vertex_count) override
	{
		m_vertex_count         = vertex_count;
		if (m_vertices.size() != vertex_count) {
			m_vertices.resize(vertex_count);
		}
		if (m_has_normal) {
			if (m_normals.size() != vertex_count) {
				m_normals.resize(vertex_count);
			}
		}

		if (vertex_count == 0) {
			m_has_normal = false;
		}
	}
	virtual HKArrayVec3*    HK_API getVertices() const override
	{
		// 参照カウント1
		auto res = HKArrayVec3::create();
		res->setCount(m_vertex_count);
		for (auto i = 0; i < m_vertex_count; ++i) {
			res->setValue(i, m_vertices[i]);
		}
		return res;
	}
	virtual void            HK_API setVertices(const HKArrayVec3* vertices) override
	{
		if (!vertices) { m_vertices.clear(); m_vertex_count = 0; return; }
		auto new_vertex_count = vertices->getCount();
		m_vertices.resize(new_vertex_count);
		for (auto i = 0; i < new_vertex_count; ++i) {
			m_vertices[i] = vertices->getValue(i);
		}
		m_vertex_count        = new_vertex_count;
	}
	virtual HKArrayVec3*    HK_API getNormals() const override
	{
		// 参照カウント1
		auto res = HKArrayVec3::create();
		if (m_has_normal) {
			res->setCount(m_vertex_count);
			for (auto i = 0; i < m_normals.size(); ++i) {
				res->setValue(i, m_normals[i]);
			}
		}
		return res;
	}
	virtual void            HK_API setNormals(const HKArrayVec3* normals) override
	{
		if (!normals) { 
			setVertexCount(0);
			return; 
		}
		auto new_vertex_count = normals->getCount();
		m_has_normal          = true;
		setVertexCount(new_vertex_count);
		for (auto i = 0; i < new_vertex_count; ++i) {
			m_normals[i]  = normals->getValue(i);
		}
	}
	virtual HKBool HK_API hasNormal() const override
	{
		return m_has_normal;
	}
	HKMeshTopology HK_API getTopologoy(HKU32 submesh_idx) const override
	{
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getTopologoy();
		}
		else {
			return HKMeshTopologyTriangles;
		}
	}
	virtual HKAabb          HK_API getAabb() const override { return HKAabb(); }
	virtual void            HK_API destroyObject() override {
		for (auto& m_submesh : m_submeshes) {
			if (m_submesh) {
				m_submesh->release();
			}
		}
		m_submeshes.clear();
	}

	HKU32                   m_vertex_count;
	HKBool                  m_has_normal;
	std::vector<HKSubMesh*> m_submeshes;
	std::vector<HKVec3>     m_vertices;
	std::vector<HKVec3>     m_normals;



};

HK_EXTERN_C HK_DLL       HKMesh* HK_API HKSubMesh_internal_getMesh(HKSubMesh* mesh)
{
	if (mesh) {
		return mesh->internal_getMesh();
	}
	else {
		return nullptr;
	}

}
HK_EXTERN_C HK_DLL const HKMesh* HK_API HKSubMesh_internal_getMesh_const(const HKSubMesh * mesh)
{
	if (mesh) {
		return mesh->internal_getMesh_const();
	}
	else {
		return nullptr;
	}
}

HK_EXTERN_C HK_DLL HKMeshTopology HK_API HKSubMesh_getTopology(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getTopologoy(); } else { return HKMeshTopologyTriangles; }
}

HK_EXTERN_C HK_DLL       HKMesh* HK_API HKMesh_create()
{
	auto res = new HKMeshImpl(); res->addRef(); return res;
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKMesh_getSubMeshCount(const HKMesh* mesh)
{
	if (mesh) {
		return mesh->getSubMeshCount();
	}
	else {
		return 0;
	}
}

HK_EXTERN_C HK_DLL HKArraySubMesh* HK_API HKMesh_getSubMeshes( HKMesh* mesh)
{
	if (mesh) {
		return mesh->getSubMeshes();
	}
	else {
		return 0;
	}
}

HK_EXTERN_C HK_DLL HKMeshTopology HK_API HKMesh_getTopologoy(const HKMesh* mesh, HKU32 submesh_idx)
{
	if (mesh) {
		return mesh->getTopologoy(submesh_idx);
	}
	else {
		return HKMeshTopologyTriangles;
	}
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKMesh_getVertexCount(const HKMesh* mesh)
{
	if (mesh) { return mesh->getVertexCount(); } else { return 0;}
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setVertexCount(HKMesh* mesh, HKU32 cnt)
{
	if (mesh) { return mesh->setVertexCount(cnt); }
}

HK_EXTERN_C HK_DLL HKArrayVec3* HK_API HKMesh_getVertices(const HKMesh* mesh)
{
	if (mesh) { return mesh->getVertices(); }
	else { return nullptr; }
}

HK_EXTERN_C HK_DLL void  HK_API HKMesh_setVertices(HKMesh* mesh, const HKArrayVec3* vertices)
{
	if (mesh) { return mesh->setVertices(vertices); }
}

HK_EXTERN_C HK_DLL HKArrayVec3* HK_API HKMesh_getNormals(const HKMesh* mesh)
{
	if (mesh) { return mesh->getNormals(); }
	else { return nullptr; }
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setNormals(HKMesh* mesh, const HKArrayVec3* normals)
{
	if (mesh) { return mesh->setNormals(normals); }
}

HK_EXTERN_C HK_DLL HKBool HK_API HKMesh_hasNormal(const HKMesh* mesh)
{
	if (mesh) { return mesh->hasNormal(); }
	else { return false; }
}

HK_SHAPE_ARRAY_IMPL_DEFINE(Mesh);
HK_SHAPE_ARRAY_IMPL_DEFINE(SubMesh);
