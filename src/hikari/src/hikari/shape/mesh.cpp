#define HK_DLL_EXPORT
#include <hikari/shape/mesh.h>
#include <hikari/ref_cnt_object.h>
#include <unordered_set>
#include <vector>

struct HK_DLL HKSubMeshImpl : public HKSubMesh, protected HKRefCntObject {
	HKSubMeshImpl(HKMesh* base) noexcept :HKSubMesh(), HKRefCntObject(), 
		mesh{ base }, 
		topology{HKMeshTopologyTriangles} ,
		aabb{},
		indices{},
		vertex_count{},
		index_count{},
		first_vertex{},
		base_vertex{}
	{}
	virtual HK_API ~HKSubMeshImpl() {}

	static auto create(HKMesh* base) -> HKSubMeshImpl* { auto res = new HKSubMeshImpl(base); res->addRef(); return res; }
	virtual void           HK_API clear() override {
		indices.clear();
		topology     = HKMeshTopologyTriangles;
		aabb         = HKAabb{};
		vertex_count = 0;
		index_count  = 0;
		first_vertex = 0;
		base_vertex  = 0;
	}
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
	virtual void           HK_API setIndices(const HKArrayU32* indices_, HKMeshTopology topology_, HKU32 base_vertex_, HKBool calc_bounds) override {
		if (!indices_ || indices_->isEmpty()) {
			clear();
			return;
		}
		topology    = topology_;
		base_vertex = base_vertex_;

		auto topology_count = HKMeshTopology_getVertexCount(topology_);
		index_count = ((indices_->getCount() + topology_count -1)/ topology_count) * topology_count;
		indices.resize(index_count);

		for (HKU32 i = 0; i < indices_->getCount(); ++i) {
			indices[i] = indices_->getValue(i);
		}

		if (calc_bounds) {
			HKAabb new_aabb;
			for (HKU32 i = 0; i < indices.size(); ++i) {
				new_aabb = new_aabb.addPoint(mesh->getVertex(indices[i] + base_vertex_));
			}
			aabb = new_aabb;
		}

		auto index_set = std::unordered_set<HKU32>(std::begin(indices), std::end(indices));
		auto index_min =*std::min_element(std::begin(index_set), std::end(index_set));

		first_vertex = index_min;
		vertex_count = index_set.size();
	}
	virtual HKArrayU32*    HK_API getIndices(HKBool add_base_vertex)     const override {		// �Q�ƃJ�E���g1
		auto res = HKArrayU32::create();
		res->setCount(index_count);
		auto index_off = add_base_vertex ? base_vertex : 0;
		for (auto i = 0; i < index_count; ++i) {
			res->setValue(i, indices[i]+ index_off);
		}
		return res;
	}
	virtual HKU32          HK_API getVertexCount() const override { return vertex_count; }
	virtual HKU32          HK_API getIndexCount()  const override { return index_count;  }
	virtual HKU32          HK_API getBaseVertex()  const override { return base_vertex;  }
	virtual HKU32          HK_API getFirstVertex() const override { return first_vertex; }
	virtual HKMeshTopology HK_API getTopology() const override
	{
		return topology;
	}
	virtual void           HK_API destroyObject() override { }
	virtual HKAabb         HK_API getAabb() const override { return aabb; }	
	virtual void           HK_API updateAabb() override {
		HKAabb new_aabb;
		for (HKU32 i = 0; i < indices.size(); ++i) {
			new_aabb = new_aabb.addPoint(mesh->getVertex(indices[i] + base_vertex));
		}
		aabb = new_aabb;
	}

	HKMesh*            mesh;
	std::vector<HKU32> indices;
	HKMeshTopology     topology;
	HKAabb             aabb;
	HKU32              vertex_count;
	HKU32              index_count;
	HKU32              first_vertex;
	HKU32              base_vertex;
};
struct HK_DLL HKMeshImpl : public HKMesh, protected HKRefCntObject {
	HKMeshImpl() noexcept : 
		HKMesh(), 
		HKRefCntObject(),
		m_submeshes{ HKSubMeshImpl::create(this)}, 
		m_vertex_count{ 0 }, 
		m_has_normal { false },
		m_has_tangent{ false },
		m_has_color  { false },
		m_has_uvs    {},
		m_vertices{},
		m_normals {},
		m_tangents{},
		m_colors{},
		m_uvs{}
	{}
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
	virtual void            HK_API clear() override {
		m_vertices.clear(); 
		m_normals .clear();
		m_tangents.clear();
		m_colors  .clear();
		for (HKU32 i = 0; i < 8; ++i) {
			m_uvs[i].clear();
		}
		m_vertex_count = 0;
	}
	virtual HKMesh*         HK_API clone() const override {
		auto res = new HKMeshImpl();
		res->addRef();
		res->setSubMeshCount(m_submeshes.size());
		{
			auto vertices = this->getVertices();
			res->setVertices(vertices);
			vertices->release();
		}
		{
			auto normals = this->getNormals();
			res->setNormals(normals);
			normals->release();
		}
		{
			auto tangents = this->getTangents();
			res->setTangents(tangents);
			tangents->release();
		}
		{
			auto colors = this->getColors();
			res->setColors(colors);
			colors->release();
		}
		for (auto i = 0; i < 8; ++i) {
			auto uv = this->getUVs(i);
			res->setUVs(i, uv);
			uv->release();
		}
		for (auto i = 0; i < getSubMeshCount(); ++i) {
			auto indices = this->getSubMeshIndices(i, false);
			res->setSubMeshIndices(i, indices, this->getSubMeshTopology(i), this->getSubMeshBaseVertex(i), true);
			indices->release();
		}
		return res;
	}
	virtual void            HK_API copy(const HKMesh* mesh) override {
		if (!mesh) {
			clear();
			return;
		}
		setSubMeshCount(mesh->getSubMeshCount());
		{
			auto vertices = mesh->getVertices();
			setVertices(vertices);
			vertices->release();
		}
		{
			auto normals = mesh->getNormals();
			setNormals(normals);
			normals->release();
		}
		{
			auto tangents = mesh->getTangents();
			setTangents(tangents);
			tangents->release();
		}
		{
			auto colors = mesh->getColors();
			setColors(colors);
			colors->release();
		}
		for (auto i = 0; i < 8; ++i) {
			auto uv = mesh->getUVs(i);
			setUVs(i, uv);
			uv->release();
		}
		for (auto i = 0; i < getSubMeshCount(); ++i) {
			auto indices = mesh->getSubMeshIndices(i, false);
			setSubMeshIndices(i, indices,mesh->getSubMeshTopology(i), mesh->getSubMeshBaseVertex(i), true);
			indices->release();
		}
	}
	virtual HKU32           HK_API getSubMeshCount() const override { return m_submeshes.size(); }
	virtual void            HK_API setSubMeshCount(HKU32 c)override {
		if (m_submeshes.size() == c) { return; }
		if (m_submeshes.size() >  c) {
			for (auto i = c; i < m_submeshes.size(); ++i) {
				m_submeshes[i]->release();
			}
			m_submeshes.resize(c);
		}
		else {
			auto offset = m_submeshes.size();
			m_submeshes.resize(c);
			for (auto i = offset; i < c; ++i) {
				m_submeshes[i] = new HKSubMeshImpl(this);
				m_submeshes[i]->addRef();
			}
		}
	}
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
		if (m_vertex_count == vertex_count) { return; }
		internal_resize(vertex_count);
		m_vertex_count = vertex_count;
	}
	virtual HKArrayVec3*    HK_API getVertices() const override
	{
		// �Q�ƃJ�E���g1
		auto res = HKArrayVec3::create();
		res->setCount(m_vertex_count);
		for (auto i = 0; i < m_vertex_count; ++i) {
			res->setValue(i, m_vertices[i]);
		}
		return res;
	}
	virtual void            HK_API setVertices(const HKArrayVec3* vertices) override
	{
		if (!vertices || vertices->isEmpty()) { clear(); return; }
		auto new_vertex_count = vertices->getCount();
		internal_resize(new_vertex_count);
		for (auto i = 0; i < new_vertex_count; ++i) {
			m_vertices[i] = vertices->getValue(i);
		}
	}
	virtual HKVec3          HK_API getVertex(HKU32 idx)const override {
		if (idx < m_vertex_count) {
			return m_vertices[idx];
		}
		else {
			return HKVec3();
		}
	}
	virtual HKArrayVec3*    HK_API getNormals() const override
	{
		// �Q�ƃJ�E���g1
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
		if (!normals || normals->isEmpty()) { m_has_normal = false; m_normals.resize(0); return; }
		auto new_vertex_count = normals->getCount();
		m_has_normal          = true;
		m_normals.resize(m_vertex_count);
		for (auto i = 0; i < std::min<HKU32>(new_vertex_count,m_vertex_count); ++i) {
			m_normals[i]  = normals->getValue(i);
		}
	}
	virtual HKBool          HK_API hasNormal() const override
	{
		return m_has_normal;
	}
	virtual HKArrayVec4*    HK_API getTangents()const override
	{
		// �Q�ƃJ�E���g1
		auto res = HKArrayVec4::create();
		if (m_has_tangent) {
			res->setCount(m_vertex_count);
			for (auto i = 0; i < m_tangents.size(); ++i) {
				res->setValue(i, m_tangents[i]);
			}
		}
		return res;
	}
	virtual void            HK_API setTangents(const HKArrayVec4* tangents) override
	{
		if (!tangents || tangents->isEmpty()) { m_has_tangent = false; m_tangents.resize(0); return; }
		auto new_vertex_count = tangents->getCount();
		m_has_tangent = true;
		m_tangents.resize(m_vertex_count);
		for (auto i = 0; i < std::min<HKU32>(new_vertex_count, m_vertex_count); ++i) {
			m_tangents[i] = tangents->getValue(i);
		}
	}
	virtual HKBool          HK_API hasTangent()const override 
	{
		return m_has_tangent;
	}
	virtual HKArrayColor*   HK_API getColors()const override {
		auto res = HKArrayColor::create();
		if (m_has_color) {
			res->setCount(m_vertex_count);
			for (auto i = 0; i < m_colors.size(); ++i) {
				res->setValue(i, m_colors[i]);
			}
		}
		return res;
	}
	virtual void            HK_API setColors(const HKArrayColor* colors) override {
		if (!colors || colors->isEmpty()) { m_has_color = false; m_colors.resize(0); return; }
		auto new_vertex_count = colors->getCount();
		m_has_color = true;
		m_colors.resize(m_vertex_count);
		for (auto i = 0; i < std::min<HKU32>(new_vertex_count, m_vertex_count); ++i) {
			m_colors[i] = colors->getValue(i);
		}
	}
	virtual HKArrayColor8*  HK_API getColor8s()const override {
		auto res = HKArrayColor8::create();
		if (m_has_color) {
			res->setCount(m_vertex_count);
			for (auto i = 0; i < m_colors.size(); ++i) {
				res->setValue(i, m_colors[i].toColor8());
			}
		}
		return res;
	}
	virtual void            HK_API setColor8s(const HKArrayColor8* colors) override {
		if (!colors || colors->isEmpty()) { m_has_color = false; m_colors.resize(0); return; }
		auto new_vertex_count = colors->getCount();
		m_has_color = true;
		m_colors.resize(m_vertex_count);
		for (auto i = 0; i < std::min<HKU32>(new_vertex_count, m_vertex_count); ++i) {
			m_colors[i] = colors->getValue(i).toColor();
		}
	}
	virtual HKBool          HK_API hasColor()const override { return m_has_color; }
	virtual HKBool          HK_API hasUV(HKU32 idx)const override {
		if (idx >= 8) { return false; }
		return m_has_uvs[idx];
	}
	virtual HKArrayVec2*    HK_API getUVs(HKU32 idx)const override {
		// �Q�ƃJ�E���g1
		auto res = HKArrayVec2::create();
		if (idx >= 8) { return res; }
		if (!m_has_uvs[idx]) {
			return res;
		}
		{
			res->setCount(m_vertex_count);
			for (auto i = 0; i < m_uvs[idx].size(); ++i) {
				res->setValue(i, m_uvs[idx][i]);
			}
		}
		return res;
	}
	virtual void            HK_API setUVs(HKU32 idx, const HKArrayVec2* uv) override {
		if (idx >= 8) { return; }
		if (!uv || uv->isEmpty()) { m_has_uvs[idx] = false; m_uvs[idx].resize(0); return; }
		auto new_vertex_count = uv->getCount();
		m_has_uvs[idx] = true;
		m_uvs[idx].resize(m_vertex_count);
		for (auto i = 0; i < std::min<HKU32>(new_vertex_count, m_vertex_count); ++i) {
			m_uvs[idx][i] = uv->getValue(i);
		}
	}
	virtual HKArrayU32*     HK_API getIndices(HKBool add_base_vertex) const override {
		if (m_submeshes.size() == 0) {
			return HKArrayU32::create();
		}
		if (m_submeshes.size() == 1) {
			return m_submeshes[0]->getIndices(add_base_vertex);
		}
		else {
			HKU32 count = 0;
			for (auto i = 0; i < m_submeshes.size(); ++i) {
				count += m_submeshes[i]->getIndexCount();
			}
			auto res = HKArrayU32::create();
			res->resize(count);

			HKU32 offset = 0;
			for (auto i = 0; i < m_submeshes.size(); ++i) {
				auto submesh_indices   = m_submeshes[i]->getIndices(add_base_vertex);
				for (auto j = 0; j < submesh_indices->getCount(); ++i) {
					res->setValue(offset + j, submesh_indices->getValue(j));
				}
				offset += m_submeshes[i]->getIndexCount();

				submesh_indices->release();
			}

			return res;
		}
	}
	virtual HKArrayU32*     HK_API getSubMeshIndices(HKU32 submesh_idx, HKBool add_base_vertex)  const override {
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getIndices(add_base_vertex);
		}
		else {
			return HKArrayU32::create();
		}
	}
	virtual void            HK_API setSubMeshIndices(HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds) override {
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->setIndices(indices,topology,base_vertex,calc_bounds);
		}
	}
	virtual HKMeshTopology  HK_API getSubMeshTopology(HKU32 submesh_idx) const override {
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getTopology();
		}
		else {
			return HKMeshTopologyTriangles;
		}
	}
	virtual HKU32           HK_API getSubMeshIndexCount(HKU32 submesh_idx) const override {
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getIndexCount();
		}
		else {
			return 0;
		}
	}
	virtual HKU32           HK_API getSubMeshBaseVertex(HKU32 submesh_idx) const override {
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getBaseVertex();
		}
		else {
			return 0;
		}
	}
	virtual HKAabb          HK_API getAabb() const override { 
		HKAabb aabb;
		for (auto& submesh : m_submeshes) {
			aabb = aabb | submesh->getAabb();
		}
		return aabb;
	}
	virtual void            HK_API updateAabb()  override {
		for (auto& submesh : m_submeshes) {
			submesh->updateAabb();
		}
	}
	virtual void            HK_API destroyObject() override {
		for (auto& m_submesh : m_submeshes) {
			if (m_submesh) {
				m_submesh->release();
			}
		}
		m_submeshes.clear();
	}

	void internal_resize(HKU32 new_vertex_count) {
		m_vertices.resize(new_vertex_count);

		if (m_has_normal)
			m_normals.resize(new_vertex_count);
		if (m_has_tangent)
			m_tangents.resize(new_vertex_count);
		if (m_has_color)
			m_colors.resize(new_vertex_count);

		for (HKU32 i = 0; i < 8; ++i) {
			if (m_has_uvs[i]) {
				m_uvs[i].resize(new_vertex_count);
			}
		}
		m_vertex_count = new_vertex_count;
	}

	HKU32                   m_vertex_count;
	HKBool                  m_has_normal;
	HKBool                  m_has_tangent;
	HKBool                  m_has_color;
	HKBool                  m_has_uvs[8];
	std::vector<HKSubMesh*> m_submeshes;
	std::vector<HKVec3>     m_vertices;
	std::vector<HKVec3>     m_normals;
	std::vector<HKVec4>     m_tangents;
	std::vector<HKColor>    m_colors;
	std::vector<HKVec2>     m_uvs[8];
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

HK_EXTERN_C HK_DLL void HK_API HKSubMesh_clear(HKSubMesh* mesh)
{
	if (mesh) { return mesh->clear(); }
}

HK_EXTERN_C HK_DLL void HK_API HKSubMesh_setIndices(HKSubMesh* mesh, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds)
{
	if (mesh) {
		mesh->setIndices(indices, topology, base_vertex, calc_bounds);
	}
}

HK_EXTERN_C HK_DLL HKArrayU32* HK_API HKSubMesh_getIndices(const HKSubMesh* mesh, HKBool add_base_vertex)
{
	if (mesh) { return mesh->getIndices(add_base_vertex); }
	else { return HKArrayU32::create(); }
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKSubMesh_getVertexCount(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getVertexCount(); }
	else { return 0; }
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKSubMesh_getIndexCount(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getIndexCount(); }
	else { return 0; }
}


HK_EXTERN_C HK_DLL HKMeshTopology HK_API HKSubMesh_getTopology(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getTopology(); }
	else { return HKMeshTopologyTriangles; }
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKSubMesh_getBaseVertex(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getBaseVertex(); }
	else { return 0; }
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKSubMesh_getFirstVertex(const HKSubMesh* mesh)
{
	if (mesh) { return mesh->getFirstVertex(); }
	else { return 0; }
}

HK_EXTERN_C HK_DLL       HKMesh* HK_API HKMesh_create()
{
	auto res = new HKMeshImpl(); res->addRef(); return res;
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_clear(HKMesh* mesh)
{
	if (mesh) { mesh->clear(); }
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

HK_EXTERN_C HK_DLL void HK_API HKMesh_setSubMeshCount(HKMesh* mesh, HKU32 count)
{
	if (mesh) { mesh->setSubMeshCount(count); }
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

HK_EXTERN_C HK_DLL HKMeshTopology HK_API HKMesh_getSubMeshTopology(const HKMesh* mesh, HKU32 submesh_idx)
{
	if (mesh) {
		return mesh->getSubMeshTopology(submesh_idx);
	}
	else {
		return HKMeshTopologyTriangles;
	}
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKMesh_getSubMeshIndexCount(const HKMesh* mesh, HKU32 submesh_idx)
{
	if (mesh) { return mesh->getSubMeshIndexCount(submesh_idx); }
	else { return 0; }
}

HK_EXTERN_C HK_DLL HKU32 HK_API HKMesh_getSubMeshBaseVertex(const HKMesh* mesh, HKU32 submesh_idx)
{
	if (mesh) { return mesh->getSubMeshBaseVertex(submesh_idx); }
	else { return 0; }
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

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKMesh_getVertex(const HKMesh* mesh, HKU32 idx)
{
	if (mesh) { return mesh->getVertex(idx); }
	else { return HKVec3(); }
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

HK_EXTERN_C HK_DLL HKArrayVec4* HK_API HKMesh_getTangents(const HKMesh* mesh)
{
	if (mesh) { return mesh->getTangents(); }
	else {
		return nullptr;
	}
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setTangents(HKMesh* mesh, const HKArrayVec4* tangents)
{
	if (mesh) { return mesh->setTangents(tangents); }
}

HK_EXTERN_C HK_DLL HKBool HK_API HKMesh_hasTangent(const HKMesh* mesh)
{
	if (mesh) { return mesh->hasTangent(); }
	else {
		return false;
	}
}

HK_EXTERN_C HK_DLL HKArrayColor* HK_API HKMesh_getColors(const HKMesh* mesh)
{
	if (mesh) { return mesh->getColors(); }
	else { return nullptr; }
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setColors(HKMesh* mesh, const HKArrayColor* colors)
{
	if (mesh) { return mesh->setColors(colors); }
}

HK_EXTERN_C HK_DLL HKArrayColor8* HK_API HKMesh_getColor8s(const HKMesh* mesh)
{
	if (mesh) { return mesh->getColor8s(); }
	else { return nullptr; }
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setColor8s(HKMesh* mesh, const HKArrayColor8* colors)
{
	if (mesh) { return mesh->setColor8s(colors); }
}

HK_EXTERN_C HK_DLL HKBool HK_API HKMesh_hasColor(const HKMesh* mesh)
{
	if (mesh) { return mesh->hasColor(); }
	else { return false; }
}

HK_EXTERN_C HK_DLL HKBool HK_API HKMesh_hasUV(const HKMesh* mesh, HKU32 idx)
{
	if (mesh) { return mesh->hasUV(idx); }
	else {
		return false;
	}
}

HK_EXTERN_C HK_DLL HKArrayVec2* HK_API HKMesh_getUVs(const HKMesh* mesh, HKU32 idx)
{
	if (mesh) { return mesh->getUVs(idx); }
	else {
		return nullptr;
	}
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setUVs(HKMesh* mesh, HKU32 idx, const HKArrayVec2* uv)
{
	if (mesh) { return mesh->setUVs(idx, uv); }
}

HK_EXTERN_C HK_DLL HKArrayU32* HK_API HKMesh_getIndices(const HKMesh* mesh, HKBool add_base_vertex)
{
	if (mesh) {
		return mesh->getIndices(add_base_vertex);
	}
	else {
		return nullptr;
	}
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_updateAabb(HKMesh* mesh)
{
	if (mesh) { mesh->updateAabb(); }
}

HK_EXTERN_C HK_DLL HKArrayU32* HK_API HKMesh_getSubMeshIndices(const HKMesh* mesh, HKU32 submesh_idx, HKBool add_base_vertex)
{
	if (mesh) {
		return mesh->getSubMeshIndices(submesh_idx,add_base_vertex);
	}
	else {
		return nullptr;
	}
}

HK_EXTERN_C HK_DLL void HK_API HKMesh_setSubMeshIndices(HKMesh* mesh, HKU32 submesh_idx, const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds)
{
	if (mesh) {
		mesh->setSubMeshIndices(submesh_idx, indices, topology, base_vertex, calc_bounds);
	}
}

HK_SHAPE_ARRAY_IMPL_DEFINE(Mesh);
HK_SHAPE_ARRAY_IMPL_DEFINE(SubMesh);

