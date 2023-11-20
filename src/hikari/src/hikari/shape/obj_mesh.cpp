#define HK_DLL_EXPORT 
#include <hikari/shape/obj_mesh.h>

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <tiny_obj_loader.h>

#include <hikari/ref_cnt_object.h>
#include <hikari/stl_utils.h>

struct HKObjSubMeshImpl : public HKObjSubMesh, protected HKRefCntObject {
	HKObjSubMeshImpl(
		HKObjMesh*                mesh,
		const std::string&        name,
		HKMeshTopology            topology,
		const std::vector<HKVec3>&vertices,
		const std::vector<HKU32>& indices,
		HKU32                     base_vertex
	) :
		HKObjSubMesh(), 
		HKRefCntObject(),
		m_mesh       { mesh},
		m_name       { name},
		m_aabb       {},
		m_topology   { topology},
		m_indices    { indices     },
		m_base_vertex{ base_vertex },
		m_index_count{(HKU32)indices.size()}
	{
		auto index_set = std::unordered_set<HKU32>(indices.begin(), indices.end());
		m_first_vertex = *std::min_element(std::begin(index_set), std::end(index_set));
		m_vertex_count = index_set.size();

		for (auto i = 0; i < indices.size(); ++i) {
			m_aabb = m_aabb.addPoint(vertices[indices[i] + base_vertex]);
		}
	}
	virtual HK_API ~HKObjSubMeshImpl() {}

	HKU32          HK_API addRef () override { return HKRefCntObject::addRef() ; }
	HKU32          HK_API release() override { return HKRefCntObject::release(); }
	HKBool         HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_SubMesh || iid == HK_OBJECT_TYPEID_ObjSubMesh) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	void           HK_API clear() override
	{
		return;
	}
	void           HK_API setIndices(const HKArrayU32* indices, HKMeshTopology topology, HKU32 base_vertex, HKBool calc_bounds) override { return; }
	HKArrayU32*    HK_API getIndices(HKBool add_base_vertex) const override
	{
		HKArrayU32* res = HKArrayU32_create();
		res->setCount(m_indices.size());
		for (auto i = 0; i < m_indices.size(); ++i) {
			res->setValue(i, m_indices[i] + add_base_vertex?m_base_vertex:0);
		}
		return res;
	}
	HKU32          HK_API getVertexCount() const override
	{
		return m_vertex_count;
	}
	HKU32          HK_API getIndexCount() const override
	{
		return m_index_count;
	}
	HKMeshTopology HK_API getTopology() const override
	{
		return m_topology;
	}
	HKU32          HK_API getBaseVertex() const override
	{
		return m_base_vertex;
	}
	HKU32          HK_API getFirstVertex() const override
	{
		return m_first_vertex;
	}
	HKMesh*        HK_API internal_getMesh() override
	{
		return m_mesh;
	}
	const HKMesh*  HK_API internal_getMesh_const() const override
	{
		return m_mesh;
	}
	void           HK_API updateAabb() override
	{
		return;
	}
	HKCStr         HK_API getName() const override
	{
		return m_name.c_str();
	}
	void           HK_API destroyObject() override
	{
	}
	HKAabb         HK_API getAabb() const override { return m_aabb; }
private:
	HKObjMesh*         m_mesh;
	std::string        m_name;
	std::vector<HKU32> m_indices;
	HKMeshTopology     m_topology;
	HKAabb             m_aabb;
	HKU32              m_vertex_count;
	HKU32              m_index_count;
	HKU32              m_first_vertex;
	HKU32              m_base_vertex;

};
struct HKObjMeshImpl : public HKObjMesh, protected HKRefCntObject {
	HKObjMeshImpl(const HKObjMesh* objmesh):
		HKObjMesh(),
		HKRefCntObject(),
		m_submeshes   {},
		m_vertex_count{ objmesh->getVertexCount()},
		m_has_normal  { objmesh->hasNormal()  },
		m_has_tangent { objmesh->hasTangent() },
		m_has_color   { objmesh->hasColor()   },
		m_has_uvs     { objmesh->hasUV(0)     },
		m_aabb        { objmesh->getAabb()    },
		m_vertices    { HKSTLUtils_toSTLArray(objmesh->getVertices())},
		m_normals     { HKSTLUtils_toSTLArray(objmesh->getNormals()) },
		m_tangents    { HKSTLUtils_toSTLArray(objmesh->getTangents())},
		m_colors      { HKSTLUtils_toSTLArray(objmesh->getColors())  },
		m_uvs         { HKSTLUtils_toSTLArray(objmesh->getUV0s())    },
		m_filename    { objmesh->getFilename()}
	{
		m_submeshes.resize(objmesh->getSubMeshCount());
		for (auto i = 0; i < m_submeshes.size(); ++i) {
			m_submeshes[i] = new HKObjSubMeshImpl(
				this, 
				objmesh->getSubMeshName(i),
				objmesh->getSubMeshTopology(i), 
				m_vertices,
				HKSTLUtils_toSTLArray(objmesh->getSubMeshIndices(i,false)),
				objmesh->getSubMeshBaseVertex(i)
			);
			m_submeshes[i]->addRef();
		}
	}
	HKObjMeshImpl() noexcept :
		HKObjMesh(),
		HKRefCntObject(),
		m_submeshes{},
		m_aabb{},
		m_vertex_count{ 0 },
		m_has_normal{ false },
		m_has_tangent{ false },
		m_has_color{ false },
		m_has_uvs{0},
		m_vertices{},
		m_normals{},
		m_tangents{},
		m_colors{},
		m_uvs{},
		m_filename{""}
	{}
	virtual HK_API ~HKObjMeshImpl() {}
	virtual HKU32           HK_API addRef()  override { return HKRefCntObject::addRef(); }
	virtual HKU32           HK_API release() override { return HKRefCntObject::release(); }
	virtual HKBool          HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_Mesh || iid == HK_OBJECT_TYPEID_ObjMesh) {
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	virtual void            HK_API clear() override {
		m_filename    = "";
		m_has_color   = false;
		m_has_normal  = false;
		m_has_tangent = false;
		m_vertices.clear();
		m_normals.clear();
		m_tangents.clear();
		m_colors.clear();
		m_aabb = HKAabb();
		for (HKU32 i = 0; i < 8; ++i) {
			m_has_uvs[i] = false;
			m_uvs[i].clear();
		}
		m_vertex_count = 0;
		for (HKU32 i = 0; i < m_submeshes.size(); ++i) {
			m_submeshes[i]->release();
		}
		m_submeshes.clear();
	}
	virtual HKMesh*         HK_API clone() const override {
		auto res = new HKObjMeshImpl(this);
		res->addRef();
		return res;
	}
	virtual void            HK_API copy(const HKMesh* mesh) override {
		if (!mesh) {
			return;
		}

	}
	virtual HKU32           HK_API getSubMeshCount() const override { return m_submeshes.size(); }
	virtual void            HK_API setSubMeshCount(HKU32 c)override {
		return;
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
		return;
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
		return;
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
		return;
	}
	virtual HKBool          HK_API hasNormal() const override
	{
		return m_has_normal;
	}
	virtual HKArrayVec4*    HK_API getTangents()const override
	{
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
		return;
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
		return;
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
		return;
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
		return;
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
				auto submesh_indices = m_submeshes[i]->getIndices(add_base_vertex);
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
		return;
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
		return m_aabb;
	}
	virtual void            HK_API updateAabb()  override {
		return;
	}
	virtual void            HK_API destroyObject() override {
		for (auto& m_submesh : m_submeshes) {
			if (m_submesh) {
				m_submesh->release();
			}
		}
		m_submeshes.clear();
	}

	void   HK_API setFilename(HKCStr filename) override
	{
		loadFile(filename);
	}
	HKCStr HK_API getFilename() const override
	{
		return m_filename.c_str();
	}
	HKBool HK_API loadFile(HKCStr filename) override
	{
		if (filename == m_filename) {
			return true;
		}
		tinyobj::ObjReaderConfig config;
		config.triangulate  = false;
		config.vertex_color = true;
		tinyobj::ObjReader  reader;

		struct HKObjMeshImpl_Vertex {
			HK_INLINE HKBool operator==(const HKObjMeshImpl_Vertex& rhs) const noexcept {
				return  HKVec3_equal_withEps(&position,&rhs.position, 1e-6f) && 
						HKVec3_equal_withEps(&normal  ,&rhs.normal  , 1e-6f) &&
						HKVec2_equal_withEps(&uv      ,&rhs.uv      , 1e-6f) &&
					    HKVec3_equal_withEps(&color   ,&rhs.color   , 1e-6f);
			}

			struct HashFunc {
				size_t operator()(const HKObjMeshImpl_Vertex& rhs) const {
					// 位置をもっとも多め, 法線とUVもそこそこ, 色は可能な限り少なめ
					// 64 bitのうち (16 -> 6 : 4 : 4 : 2)
					size_t hash_px = std::hash<float>()(rhs.position.x);//12
					size_t hash_py = std::hash<float>()(hash_px + rhs.position.y);//12
					size_t hash_pz = std::hash<float>()(hash_py + rhs.position.z);

					size_t hash_nx = std::hash<float>()(rhs.normal.x);
					size_t hash_ny = std::hash<float>()(hash_nx + rhs.normal.y);
					size_t hash_nz = std::hash<float>()(hash_ny + rhs.normal.z);

					size_t hash_tx = std::hash<float>()(rhs.uv.x);
					size_t hash_ty = std::hash<float>()(hash_tx + rhs.uv.y);

					size_t hash_cx = std::hash<float>()(rhs.color.x);
					size_t hash_cy = std::hash<float>()(hash_cx + rhs.color.y);
					size_t hash_cz = std::hash<float>()(hash_cy + rhs.color.z);

					return  (size_t(hash_pz & size_t(0x3F))) | 
						    (size_t(hash_nz & size_t(0xF)) << size_t(6)) |
						    (size_t(hash_ty & size_t(0xF)) << size_t(10))|
						    (size_t(hash_cz & size_t(0x3)) << size_t(14));
				}
			};
			// 頂点
			HKVec3 position;
			// 法線
			HKVec3 normal;
			// UV座標
			HKVec2 uv;
			// 色
			HKVec3 color;
		};

		if (reader.ParseFromFile(filename, config)) {
			clear();
			auto& attribs   = reader.GetAttrib();
			auto& shapes    = reader.GetShapes();
			auto& materials = reader.GetMaterials();

			std::unordered_map<HKObjMeshImpl_Vertex, size_t, HKObjMeshImpl_Vertex::HashFunc> vertex_map = {};
			std::vector<std::tuple<std::string, HKMeshTopology, std::vector<HKU32>>>          submeshes = {};

			for (HKU64 s = 0; s < shapes.size(); ++s) {
				HKU64 idx_offset = 0;
				std::unordered_map<HKU32, HKU32> topology_and_count_map = {};
				for (HKU64 f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
					if (shapes[s].mesh.num_face_vertices[f] >= 5) { printf("failed to support 5th polygons!\n"); continue; }
					auto iter = topology_and_count_map.find(shapes[s].mesh.num_face_vertices[f]);
					if (iter != std::end(topology_and_count_map)) {
						iter->second++;
					}
					else {
						topology_and_count_map.insert({ shapes[s].mesh.num_face_vertices[f] ,0 });
					}
				}
				std::tuple<std::string, HKMeshTopology, std::vector<HKU32>> tmp_submeshes[4] = {};
				HKBool is_multiple_topology = false;
				if (topology_and_count_map.size() > 1) {
					is_multiple_topology = true;
				}
				for (auto& [fv,count]: topology_and_count_map){
					std::string    submesh_name = shapes[s].name;
					
					HKMeshTopology submesh_topology = {};
					if (fv == 1) {
						submesh_topology = HKMeshTopologyPoints;
						if (is_multiple_topology) {
							submesh_name += ".points";
						}
					}
					if (fv == 2) {
						submesh_topology = HKMeshTopologyLines;
						if (is_multiple_topology) {
							submesh_name += ".lines";
						}
					}
					if (fv == 3) {
						submesh_topology = HKMeshTopologyTriangles;
						if (is_multiple_topology) {
							submesh_name += ".triangles";
						}
					}
					if (fv == 4) {
						submesh_topology = HKMeshTopologyQuads;
						if (is_multiple_topology) {
							submesh_name += ".quads";
						}
					}

					std::vector<HKU32> submesh_indices = {};
					submesh_indices.reserve(count);
					tmp_submeshes[fv-1] = std::make_tuple(submesh_name, submesh_topology, submesh_indices);
				}
				for (HKU64 f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
					HKU64 fv = shapes[s].mesh.num_face_vertices[f];
					// トポロジーに応じてどこにインサートするか変える
					for (HKU64 v = 0; v < fv; ++v) {
						auto& idx = shapes[s].mesh.indices[idx_offset + v];
						
						HKObjMeshImpl_Vertex tmp = {};
						tmp.position.x = attribs.GetVertices()[3 * idx.vertex_index + 0];
						tmp.position.y = attribs.GetVertices()[3 * idx.vertex_index + 1];
						tmp.position.z = attribs.GetVertices()[3 * idx.vertex_index + 2];

						if (idx.normal_index >= 0) {
							tmp.normal.x = attribs.normals[3 * idx.normal_index + 0];
							tmp.normal.y = attribs.normals[3 * idx.normal_index + 1];
							tmp.normal.z = attribs.normals[3 * idx.normal_index + 2];
						}
						else {
							tmp.normal.x = 0.0f;
							tmp.normal.y = 0.0f;
							tmp.normal.z = 1.0f;
						}
						if (idx.texcoord_index >= 0) {
							tmp.uv.x = attribs.texcoords[2 * idx.texcoord_index + 0];
							tmp.uv.y = attribs.texcoords[2 * idx.texcoord_index + 1];
						}
						else {
							tmp.uv.x = 0.0f;
							tmp.uv.y = 0.0f;
						}

						tmp.color.x = attribs.colors[3 * idx.vertex_index + 0];
						tmp.color.y = attribs.colors[3 * idx.vertex_index + 1];
						tmp.color.z = attribs.colors[3 * idx.vertex_index + 2];

						auto is_found = vertex_map.find(tmp);
						HKU64 v_index = 0;
						if (is_found == std::end(vertex_map)) {
							auto val = vertex_map.size();
							vertex_map.insert({tmp,val });
							v_index = val;
						}
						else {
							v_index = is_found->second;
						}

						std::get<std::vector<HKU32>>(tmp_submeshes[fv - 1]).push_back(v_index);
					}
					idx_offset += fv;
				}
				for (auto& [fv, count] : topology_and_count_map) {
					submeshes.push_back(tmp_submeshes[fv - 1]);
				}
			}
			std::vector<HKVec3> vertices(vertex_map.size());
			std::vector<HKVec3> normals(vertex_map.size());
			std::vector<HKVec2> uvs(vertex_map.size());
			std::vector<HKColor> colors(vertex_map.size());
			
			for (auto& [vertex, idx] : vertex_map) {
				vertices[idx] = vertex.position;
				normals[idx]  = vertex.normal;
				uvs[idx]      = vertex.uv;
				colors[idx]   = HKColor{ vertex.color.x,vertex.color.y,vertex.color.z,1.0f };
			}

			m_submeshes.reserve(submeshes.size());
			for (auto& [name, topology, indices] : submeshes) {
				auto submesh = new HKObjSubMeshImpl(this, name,topology, vertices, indices, 0);
				submesh->addRef();
				m_submeshes.push_back(submesh);
			}

			{
				auto total_indices = 0;
				for (auto& submesh : m_submeshes) {
					total_indices += submesh->getIndexCount();
				}

				printf("vertex count: %d -> %d\n", total_indices, std::size(vertices));
			}

			m_filename   = filename;
			m_vertex_count=std::size(vertices);
			m_vertices   = std::move(vertices);
			m_normals    = std::move(normals);
			m_has_normal = true;
			m_colors     = std::move(colors);
			m_has_color  = true;
			m_tangents   = {};
			m_has_tangent= false;
			m_uvs[0]     = std::move(uvs);
			m_has_uvs[0] = true;
			m_aabb = HKAabb{};
			for (auto& submesh : m_submeshes) {
				m_aabb = m_aabb| submesh->getAabb();
			}
			return true;
		}
		return false;
	}
	HKCStr HK_API getSubMeshName(HKU32 submesh_idx) const override
	{
		if (submesh_idx < m_submeshes.size()) {
			return m_submeshes[submesh_idx]->getName();
		}
		else {
			return "";
		}
	}

	std::string             m_filename;
	HKAabb                  m_aabb;
	HKU32                   m_vertex_count;
	HKBool                  m_has_normal;
	HKBool                  m_has_tangent;
	HKBool                  m_has_color;
	HKBool                  m_has_uvs[8];
	std::vector<HKObjSubMesh*> m_submeshes;
	std::vector<HKVec3>     m_vertices;
	std::vector<HKVec3>     m_normals;
	std::vector<HKVec4>     m_tangents;
	std::vector<HKColor>    m_colors;
	std::vector<HKVec2>     m_uvs[8];

};

HK_EXTERN_C HK_DLL HKCStr HK_API HKObjSubMesh_getName(const HKObjSubMesh* obj_mesh)
{
	if (obj_mesh) {
		return obj_mesh->getName();
	}
	else {
		return "";
	}
}

HK_EXTERN_C HK_DLL HKObjMesh* HK_API HKObjMesh_create()
{
	auto res = new HKObjMeshImpl();
	res->addRef();
	return res;
}

HK_EXTERN_C HK_DLL HKCStr     HK_API HKObjMesh_getFilename(const HKObjMesh* obj_mesh)
{
	if (obj_mesh) {
		return obj_mesh->getFilename();
	}
	else {
		return "";
	}
}

HK_EXTERN_C HK_DLL void       HK_API HKObjMesh_setFilename( HKObjMesh* obj_mesh, HKCStr filename)
{
	if (obj_mesh) {
		obj_mesh->setFilename(filename);
	}
}

HK_EXTERN_C HK_DLL HKBool     HK_API HKObjMesh_loadFile(HKObjMesh* obj_mesh, HKCStr filename)
{
	if (obj_mesh) { return obj_mesh->loadFile(filename); }
	else { return false; }
}

HK_EXTERN_C HK_DLL HKCStr HK_API HKObjMesh_getSubMeshName(const HKObjMesh* obj_mesh, HKU32 submesh_idx)
{
	if (obj_mesh) { return obj_mesh->getSubMeshName(submesh_idx); }
	else { return ""; }
}


HK_SHAPE_ARRAY_IMPL_DEFINE(ObjMesh);
HK_SHAPE_ARRAY_IMPL_DEFINE(ObjSubMesh);