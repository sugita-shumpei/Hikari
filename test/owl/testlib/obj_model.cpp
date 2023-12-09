#include "obj_model.h"
#include <random>
#include <utility>
#include <tiny_obj_loader.h>
#include <mikktspace.h>

struct TinyObjIndexWrapper {
	tinyobj::index_t value;
	uint32_t         tangent_idx;
	inline bool operator==(const TinyObjIndexWrapper& v) const{
		if (value.vertex_index   != v.value.vertex_index  ) { return false; }
		if (value.texcoord_index != v.value.texcoord_index) { return false; }
		if (value.normal_index   != v.value.normal_index  ) { return false; }
		if (tangent_idx != v.tangent_idx) { return false; }
		return true;
	}
};

template<> struct ::std::hash<TinyObjIndexWrapper> {
	size_t operator()(const TinyObjIndexWrapper& i) const{
		uint64_t v  = 0;
		std::int32_t arr[] = { i.value.vertex_index,i.value.texcoord_index };
		std::memcpy(&v, arr, sizeof(arr));
		uint32_t vn = static_cast<uint32_t>(i.value.normal_index);
		return std::hash<uint64_t>()(v) ^ std::hash<uint32_t>()(vn) ^ std::hash<uint32_t>()(i.tangent_idx);
	}
};

struct TinyObjMeshWrapper {
	tinyobj::ObjReader*   p_obj_reader;
	uint32_t              shape_index ;
	std::vector<uint32_t> idx_offsets ;
	std::vector<float>    tangents    ;

	void generateTangents();
	auto getAttrib()  const -> const tinyobj::attrib_t& {
		return p_obj_reader->GetAttrib();
	}
	auto getShape()   const -> const tinyobj::shape_t& {
		return p_obj_reader->GetShapes()[shape_index];
	}
	auto getNumFaces()const { return getShape().mesh.num_face_vertices.size(); }
	auto getNumVerticesOfFace(size_t face_idx)const { return getShape().mesh.num_face_vertices[face_idx]; }
	void getPosition(size_t face_idx, size_t vert_idx, float fvPosOut []) const {
		auto& shape     = getShape();
		if (shape.mesh.num_face_vertices[face_idx] <= vert_idx) { return; }
		auto vertex_idx = shape.mesh.indices[idx_offsets[face_idx]+vert_idx].vertex_index;
		fvPosOut[0u] = getAttrib().vertices[3u*vertex_idx+0u];
		fvPosOut[1u] = getAttrib().vertices[3u*vertex_idx+1u];
		fvPosOut[2u] = getAttrib().vertices[3u*vertex_idx+2u];
	}
	void getNormal  (size_t face_idx, size_t vert_idx, float fvNormOut[]) const {
		auto& shape = getShape();
		auto num_faces = shape.mesh.num_face_vertices[face_idx];
		if ( num_faces <= vert_idx) { return; }
		auto normal_idx = shape.mesh.indices[idx_offsets[face_idx] + vert_idx].normal_index;
		if (normal_idx > 0) {
			fvNormOut[0u] = getAttrib().normals[3u * normal_idx + 0u];
			fvNormOut[1u] = getAttrib().normals[3u * normal_idx + 1u];
			fvNormOut[2u] = getAttrib().normals[3u * normal_idx + 2u];
		}
		else {
			if (num_faces == 3) {
				uint32_t vertex_indices[] = {
					shape.mesh.indices[idx_offsets[face_idx] + 0].vertex_index,
					shape.mesh.indices[idx_offsets[face_idx] + 1].vertex_index,
					shape.mesh.indices[idx_offsets[face_idx] + 2].vertex_index
				};
				float vp0[] = {
					getAttrib().vertices[3u * vertex_indices[0u] + 0u],
					getAttrib().vertices[3u * vertex_indices[0u] + 1u],
					getAttrib().vertices[3u * vertex_indices[0u] + 2u]
				};
				float vp1[] = {
					getAttrib().vertices[3u * vertex_indices[1u] + 0u],
					getAttrib().vertices[3u * vertex_indices[1u] + 1u],
					getAttrib().vertices[3u * vertex_indices[1u] + 2u]
				};
				float vp2[] = {
					getAttrib().vertices[3u * vertex_indices[2u] + 0u],
					getAttrib().vertices[3u * vertex_indices[2u] + 1u],
					getAttrib().vertices[3u * vertex_indices[2u] + 2u]
				};
				float v01[] = { vp1[0] - vp0[0],vp1[1] - vp0[1],vp1[2] - vp0[2] };
				float v12[] = { vp2[0] - vp1[0],vp2[1] - vp1[1],vp2[2] - vp1[2] };
				float nfv[] = { 
					v01[1] * v12[2] - v01[2] * v12[1],
					v01[2] * v12[0] - v01[0] * v12[2],
					v01[0] * v12[1] - v01[1] * v12[0]
				};
				float inv_len = 1.0f/std::sqrtf(nfv[0] * nfv[0] + nfv[1] * nfv[1] + nfv[2] * nfv[2]);
				fvNormOut[0u] = nfv[0u] * inv_len;
				fvNormOut[1u] = nfv[1u] * inv_len;
				fvNormOut[2u] = nfv[2u] * inv_len;
			}
		}
	}
	void getTexCoord(size_t face_idx, size_t vert_idx, float       fvTexCOut[]) const {
		auto& shape = getShape();
		auto num_faces = shape.mesh.num_face_vertices[face_idx];
		if (num_faces <= vert_idx) { return; }
		auto texcoord_index = shape.mesh.indices[idx_offsets[face_idx] + vert_idx].texcoord_index;
		if ( texcoord_index > 0) {
			fvTexCOut[0u] = getAttrib().texcoords[2u * texcoord_index + 0u];
			fvTexCOut[1u] = getAttrib().texcoords[2u * texcoord_index + 1u];
		}
		else {
			fvTexCOut[0u] = 0.5f;
			fvTexCOut[1u] = 0.5f;
		}
	}
	void setTangent (size_t face_idx, size_t vert_idx, const float fvTangent[], const float fSign){
		auto& shape = getShape();
		auto num_faces = shape.mesh.num_face_vertices[face_idx];
		if (num_faces <= vert_idx) { return; }
		{
			float fvNormalOut[3];
			getNormal(face_idx, vert_idx, fvNormalOut);

			//
			if (fabsf(
				fvNormalOut[0] * fvTangent[0] +
				fvNormalOut[1] * fvTangent[1] +
				fvNormalOut[2] * fvTangent[2]
				)> 0.8f) {
				if (fabsf(fvTangent[1]) < 0.8f) {
					float* vn3   = fvNormalOut;
					float vup[3] = { 0.0f,1.0f,0.0f };
					float nfv[]  = { 
						vup[1] * vn3[2] - vup[2] * vn3[1],
						vup[2] * vn3[0] - vup[0] * vn3[2],
						vup[0] * vn3[1] - vup[1] * vn3[0]
					};
					//もしTangentと法線方向が0だったら->Tangentを変更する
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 0] = nfv[0];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 1] = nfv[1];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 2] = nfv[2];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 3] = fSign;
				}
				if (fabsf(fvTangent[2]) < 0.8f) {
					float* vn3   = fvNormalOut;
					float vup[3] = { 0.0f,0.0f,1.0f };
					float nfv[]  = {
						vup[1] * vn3[2] - vup[2] * vn3[1],
						vup[2] * vn3[0] - vup[0] * vn3[2],
						vup[0] * vn3[1] - vup[1] * vn3[0]
					};
					//もしTangentと法線方向が0だったら->Tangentを変更する
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 0] = nfv[0];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 1] = nfv[1];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 2] = nfv[2];
					tangents[4 * (idx_offsets[face_idx] + vert_idx) + 3] = fSign;
				}
			}
			else {
				//もしTangentと法線方向が0だったら->Tangentを変更する
				tangents[4 * (idx_offsets[face_idx] + vert_idx) + 0] = fvTangent[0];
				tangents[4 * (idx_offsets[face_idx] + vert_idx) + 1] = fvTangent[1];
				tangents[4 * (idx_offsets[face_idx] + vert_idx) + 2] = fvTangent[2];
				tangents[4 * (idx_offsets[face_idx] + vert_idx) + 3] = fSign;
			}
		}
	}
};

extern "C" {
	static int  TinyObjMeshWrapper_getNumFaces(const SMikkTSpaceContext* pContext) {
		return reinterpret_cast<const TinyObjMeshWrapper*>(pContext->m_pUserData)->getNumFaces();
	}
	static int  TinyObjMeshWrapper_getNumVerticesOfFace(const SMikkTSpaceContext* pContext, const int iFace) {
		return reinterpret_cast<const TinyObjMeshWrapper*>(pContext->m_pUserData)->getNumVerticesOfFace(iFace);
	}
	static void TinyObjMeshWrapper_getPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
		return reinterpret_cast<const TinyObjMeshWrapper*>(pContext->m_pUserData)->getPosition(iFace,iVert,fvPosOut);
	}
	static void TinyObjMeshWrapper_getNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
		return reinterpret_cast<const TinyObjMeshWrapper*>(pContext->m_pUserData)->getNormal(iFace, iVert, fvNormOut);
	}
	static void TinyObjMeshWrapper_getTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
		return reinterpret_cast<const TinyObjMeshWrapper*>(pContext->m_pUserData)->getTexCoord(iFace, iVert, fvTexcOut);
	}
	static void TinyObjMeshWrapper_setTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
		return reinterpret_cast<TinyObjMeshWrapper*>(pContext->m_pUserData)->setTangent(iFace, iVert, fvTangent, fSign);
	}
}

void TinyObjMeshWrapper::generateTangents() {
	tangents.clear();
	tangents.resize(getShape().mesh.indices.size()*4);
	SMikkTSpaceInterface space_interface   = {};
	space_interface.m_getNumFaces          = TinyObjMeshWrapper_getNumFaces;
	space_interface.m_getNumVerticesOfFace = TinyObjMeshWrapper_getNumVerticesOfFace;
	space_interface.m_getPosition          = TinyObjMeshWrapper_getPosition;
	space_interface.m_getNormal            = TinyObjMeshWrapper_getNormal;
	space_interface.m_getTexCoord          = TinyObjMeshWrapper_getTexCoord;
	space_interface.m_setTSpaceBasic       = TinyObjMeshWrapper_setTSpaceBasic;
	SMikkTSpaceContext space_context ={};
	space_context.m_pInterface       = &space_interface;
	space_context.m_pUserData        = this;
	genTangSpaceDefault(&space_context);
}

bool hikari::test::owl::testlib::ObjModel::load(std::string filename)
{
	if (m_filename == filename) { return true; }
	tinyobj::ObjReader       reader;
	tinyobj::ObjReaderConfig config;
	//config.triangulate = true;
	if (reader.ParseFromFile(filename, config)) {
		m_meshes.clear();
		auto& shapes = reader.GetShapes();
		
		m_meshes.reserve(shapes.size());
		auto& attrib = reader.GetAttrib();
		
		uint32_t shape_idx = 0;
		for (auto& shape : shapes) {
			auto& shape_name = shape.name;
			auto& mesh       = shape.mesh;


			std::unordered_map<TinyObjIndexWrapper, uint32_t> index_map = {};
			std::unordered_set<uint16_t> mat_idx_set(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));
			auto mat_indices = std::vector<uint8_t>();
			auto materials   = std::vector<uint16_t>(std::begin(mat_idx_set), std::end(mat_idx_set));

			std::unordered_set<uint16_t> smoothing_group_idx_set(std::begin(shape.mesh.smoothing_group_ids), std::end(shape.mesh.smoothing_group_ids));
			auto smoothing_group_indices = std::vector<uint8_t>();
			auto smoothing_groups        = std::vector<uint8_t>(std::begin(smoothing_group_idx_set), std::end(smoothing_group_idx_set));

			std::vector<uint32_t> idx_offsets(shape.mesh.indices.size(),0u);
			{
				uint32_t face_idx   = 0;
				uint32_t idx_offset = 0;
				for (auto& num_face_vertex : mesh.num_face_vertices) {
					idx_offsets[face_idx] = idx_offset;
					idx_offset += num_face_vertex;
					face_idx++;
				}
			}

			std::vector<uint32_t>                            tmp_tangent_indices = {};
			std::vector<hikari::test::owl::testlib::ObjVec4> tmp_tangents        = {};
			tmp_tangents.reserve(shape.mesh.indices.size());
			tmp_tangent_indices.reserve(shape.mesh.indices.size());
			{
				TinyObjMeshWrapper res = {};
				res.p_obj_reader = &reader;
				res.shape_index  = shape_idx;
				res.idx_offsets  = idx_offsets;
				res.generateTangents();
				// ���K������Ă���Γ�r�b�g�̒l�ŕ\���\
				std::unordered_map<uint64_t, uint32_t> tangent_xy_map;
				std::unordered_map<uint32_t, uint32_t> tangent_zw_map;
				for (auto i = 0; i < res.tangents.size()/4;++i) {
					//64bit�ɕϊ�����
					float xy[]      = { res.tangents[4 * i + 0],res.tangents[4 * i + 1] };
					uint64_t key    = 0;
					uint32_t val_xy = 0;
					std::memcpy(&key, xy, sizeof(uint64_t));
					{
						auto iter = tangent_xy_map.find(key);
						if (iter==std::end(tangent_xy_map)) {
							val_xy = tangent_xy_map.size();
							tangent_xy_map.insert({ key,val_xy });
						}
						else {
							val_xy = iter->second;
						}
					}
					{
						auto key_zw   = 2 * static_cast<int>(res.tangents[4 * i + 2] >= 0)+static_cast<int>(res.tangents[4*i+3]>0);
						auto key_xyzw = 4 * val_xy + key_zw;
						auto iter     = tangent_zw_map.find(key_xyzw);
						// �������߂Č���������
						if (iter  == std::end(tangent_zw_map)) {
							tmp_tangent_indices.push_back(tmp_tangents.size());
							tangent_zw_map.insert({ key_xyzw, tmp_tangents.size() });
							tmp_tangents.push_back({
								res.tangents[4 * i + 0],res.tangents[4 * i + 1],
								res.tangents[4 * i + 2],res.tangents[4 * i + 3]
							});
						}
						else {
							tmp_tangent_indices.push_back(iter->second);
						}
						
					}
				}

			}
			tmp_tangents.shrink_to_fit();

			std::vector<ObjVec3> positions   = {};
			std::vector<ObjVec3> normals     = {};
			std::vector<ObjVec4> tangents    = {};
			std::vector<ObjVec2> uvs         = {};
			std::vector<ObjIdx3> tri_indices = {};
			{
				uint32_t face_idx = 0;
				for (auto& num_face_vertex : mesh.num_face_vertices) {
					uint32_t idx_offset = idx_offsets[face_idx];
					// ������3�Ƃ͌���Ȃ��Ȃ�
					if (num_face_vertex == 3u) {
						uint32_t idx3[3] = {};
						for (uint32_t f = 0; f < 3; ++f) {
							auto& idx = mesh.indices[idx_offset + f];
							auto  tangent_idx = tmp_tangent_indices[idx_offset + f];
							auto iter = index_map.find(TinyObjIndexWrapper(idx, tangent_idx));
							if (iter != index_map.end()) {
								idx3[f] = iter->second;
							}
							else {
								auto v_idx = positions.size();
								auto vpx = attrib.GetVertices()[3u * idx.vertex_index + 0u];
								auto vpy = attrib.GetVertices()[3u * idx.vertex_index + 1u];
								auto vpz = attrib.GetVertices()[3u * idx.vertex_index + 2u];
								positions.push_back({ vpx,vpy,vpz });
								if (idx.normal_index > 0) {
									auto vnx = attrib.normals[3u * idx.normal_index + 0u];
									auto vny = attrib.normals[3u * idx.normal_index + 1u];
									auto vnz = attrib.normals[3u * idx.normal_index + 2u];
									normals.push_back({ vnx, vny, vnz });
								}
								else {
									normals.push_back({ 0.0f,0.0f,0.0f });
								}
								if (idx.texcoord_index > 0) {
									auto vtx = attrib.texcoords[2u * idx.texcoord_index + 0u];
									auto vty = attrib.texcoords[2u * idx.texcoord_index + 1u];
									uvs.push_back({ vtx, vty });
								}
								else {
									uvs.push_back({ 0.5f,0.5f });
								}
								{
									tangents.push_back(tmp_tangents[tangent_idx]);
								}
								idx3[f] = v_idx;
								index_map.insert({ TinyObjIndexWrapper(idx),v_idx });
							}
						}
						// material  index
						{
							auto mat_idx = shape.mesh.material_ids[face_idx];
							auto iter = std::find(std::begin(materials), std::end(materials), mat_idx);
							mat_indices.push_back(std::distance(std::begin(materials), iter));
						}
						// smoothing group
						{
							auto smo_idx = shape.mesh.smoothing_group_ids[face_idx];
							auto iter = std::find(std::begin(smoothing_groups), std::end(smoothing_groups), smo_idx);
							smoothing_group_indices.push_back(std::distance(std::begin(smoothing_groups), iter));
						}
						// index
						tri_indices.push_back({ idx3[0],idx3[1],idx3[2] });
					}
					else if (num_face_vertex == 4u) {
						uint32_t idx4[4] = {};
						for (uint32_t f = 0; f < 4; ++f) {
							auto& idx = mesh.indices[idx_offset + f];
							auto tangent_idx = tmp_tangent_indices[idx_offset + f];
							auto iter = index_map.find(TinyObjIndexWrapper(idx, tangent_idx));
							if (iter != index_map.end()) {
								idx4[f] = iter->second;
							}
							else {
								auto v_idx = positions.size();
								auto vpx = attrib.GetVertices()[3u * idx.vertex_index + 0u];
								auto vpy = attrib.GetVertices()[3u * idx.vertex_index + 1u];
								auto vpz = attrib.GetVertices()[3u * idx.vertex_index + 2u];
								positions.push_back({ vpx,vpy,vpz });
								if (idx.normal_index > 0) {
									auto vnx = attrib.normals[3u * idx.normal_index + 0u];
									auto vny = attrib.normals[3u * idx.normal_index + 1u];
									auto vnz = attrib.normals[3u * idx.normal_index + 2u];
									normals.push_back({ vnx, vny, vnz });
								}
								else {
									normals.push_back({ 0.0f,0.0f,0.0f });
								}
								if (idx.texcoord_index > 0) {
									auto vtx = attrib.texcoords[2u * idx.texcoord_index + 0u];
									auto vty = attrib.texcoords[2u * idx.texcoord_index + 1u];
									uvs.push_back({ vtx, vty });
								}
								else {
									uvs.push_back({ 0.5f,0.5f });
								}
								{
									tangents.push_back(tmp_tangents[tangent_idx]);
								}
								idx4[f] = v_idx;
								index_map.insert({ TinyObjIndexWrapper(idx),v_idx });
							}
						}
						// material  index
						{
							auto mat_idx = shape.mesh.material_ids[face_idx];
							auto iter = std::find(std::begin(materials), std::end(materials), mat_idx);
							mat_indices.push_back(std::distance(std::begin(materials), iter));
						}
						// smoothing group
						{
							auto smo_idx = shape.mesh.smoothing_group_ids[face_idx];
							auto iter = std::find(std::begin(smoothing_groups), std::end(smoothing_groups), smo_idx);
							smoothing_group_indices.push_back(std::distance(std::begin(smoothing_groups), iter));
						}
						// index
						tri_indices.push_back({ idx4[0],idx4[1],idx4[2] });
						tri_indices.push_back({ idx4[0],idx4[2],idx4[3] });
					}
					else {
						throw std::runtime_error("Failed To Support Num Face Vertex >= 5!");
					}
					face_idx++;
				}
			}

			ObjBBox m_bbox;
			for (auto& position : positions) {
				m_bbox.addPoint(position);
			}

			printf("mesh[%s].vertex_count=%d->%d\n", shape.name.c_str(),3* tri_indices.size(), positions.size());
			m_meshes.insert({ 
				shape_name,
				{ std::move(tri_indices), std::move(positions), std::move(normals),std::move(tangents), std::move(uvs), mat_indices, materials, smoothing_group_indices, smoothing_groups,m_bbox}
			});


			++shape_idx;
		}
		m_bbox = {};
		
		for (auto& [name,mesh] : m_meshes) {
			m_bbox.addBBox(mesh.bbox);
		}
		m_filename = filename;

		auto& materials = reader.GetMaterials();
		
		m_materials.clear();
		m_materials.reserve(materials.size());

		m_textures.clear();

		auto textures = m_textures;
		textures.push_back({});

		std::unordered_map<std::string, uint16_t> tex_idx_map = {};

		auto get_tex_idx = [&tex_idx_map,&textures](const std::string& name) -> uint16_t {
			if (name == "") { return 0; }
			auto iter = tex_idx_map.find(name);
			if (iter == tex_idx_map.end()) {
				auto res = textures.size();
				tex_idx_map.insert({ name,textures.size() });
				textures.push_back({ name });
				return res;
			}
			else {
				return iter->second;
			}
		};

		for (auto& material : materials) {
			auto mat             = ObjMaterial();
			mat.name             = material.name;
			mat.illum            = material.illum;
			mat.diffuse.x        = material.diffuse[0];
			mat.diffuse.y        = material.diffuse[1];
			mat.diffuse.z        = material.diffuse[2];
			mat.emission.x       = material.emission[0];
			mat.emission.y       = material.emission[1];
			mat.emission.z       = material.emission[2];
			mat.specular.x       = material.specular[0];
			mat.specular.y       = material.specular[1];
			mat.specular.z       = material.specular[2];
			mat.ior              = material.ior;
			mat.shinness         = material.shininess;
			mat.dissolve         = material.dissolve;
			mat.tex_diffuse      = get_tex_idx(material.diffuse_texname );
			mat.tex_specular     = get_tex_idx(material.specular_texname);
			mat.tex_shinness     = get_tex_idx(material.specular_highlight_texname);
			mat.tex_emission     = get_tex_idx(material.emissive_texname);
			mat.tex_alpha        = get_tex_idx(material.alpha_texname);
			mat.tex_normal       = get_tex_idx(material.bump_texname);
			mat.tex_reflection   = get_tex_idx(material.reflection_texname);
			mat.tex_normal       = get_tex_idx(material.normal_texname);
			mat.tex_displacement = get_tex_idx(material.displacement_texname);
			m_materials.push_back(mat);
		}

		m_textures = std::move(textures);

		return true;
	}
	return false;
}

void hikari::test::owl::testlib::ObjModel::setFilename(std::string filename)
{
	(void)load(filename);
}

auto hikari::test::owl::testlib::ObjMesh ::getVisSmoothColors() const -> std::vector<ObjVec3>
{
	std::vector<ObjVec3> colors(smoothing_group_indices.size());
	for (auto i = 0; i < smoothing_group_indices.size(); ++i) {
		auto smooth_group = smoothing_groups[smoothing_group_indices[i]];
		std::mt19937 mt(smooth_group);
		auto uni = std::uniform_real_distribution<float>(0.f, 1.f);
		colors[i] = { uni(mt),uni(mt),uni(mt) };
	}
	return colors;
}

auto hikari::test::owl::testlib::ObjMesh ::getVisMaterialColors() const -> std::vector<ObjVec3>
{
	std::vector<ObjVec3> colors(mat_indices.size());
	for (auto i = 0; i < mat_indices.size(); ++i) {
		auto smooth_group = materials[mat_indices[i]];
		if (smooth_group == 0) {
			colors[i] = { 0.0f,0.0f,0.0f };
		}
		else {

			std::mt19937 mt(smooth_group);
			auto uni = std::uniform_real_distribution<float>(0.f, 1.f);
			colors[i] = { uni(mt),uni(mt),uni(mt) };
		}
	}
	return colors;
}

auto hikari::test::owl::testlib::ObjMesh ::getSubMeshIndices(uint32_t mat_idx) const -> std::vector<ObjIdx3>
{
	if (mat_idx < materials.size()) {
		if (materials.size() == 1) { return tri_indices; }
		std::vector<ObjIdx3> res = {};
		res.reserve(tri_indices.size());
		auto idx = 0u;
		for (auto& mat_index : mat_indices) {
			if (mat_index == mat_idx) {
				res.push_back(tri_indices[idx]);
			}
			++idx;
		}
		res.shrink_to_fit();
		return res;
	}
	else {
		return {};
	}
}
