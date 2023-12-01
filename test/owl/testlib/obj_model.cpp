#include "obj_model.h"
#include <random>
#include <utility>
#include <tiny_obj_loader.h>

struct TinyobjIndexWrapper {
	tinyobj::index_t value;
	inline bool operator==(const TinyobjIndexWrapper& v) const{
		if (value.vertex_index   != v.value.vertex_index  ) { return false; }
		if (value.texcoord_index != v.value.texcoord_index) { return false; }
		if (value.normal_index   != v.value.normal_index  ) { return false; }
		return true;
	}
};

template<> struct ::std::hash<TinyobjIndexWrapper> {
	size_t operator()(const TinyobjIndexWrapper& i) const{
		uint64_t v  = 0;
		std::int32_t arr[] = { i.value.vertex_index,i.value.texcoord_index };
		std::memcpy(&v, arr, sizeof(arr));
		uint32_t vn = static_cast<uint32_t>(i.value.normal_index);
		return std::hash<uint64_t>()(v) ^ std::hash<uint32_t>()(vn);
	}
};

bool hikari::test::owl::testlib::ObjModel::load(std::string filename)
{
	if (m_filename == filename) { return true; }
	tinyobj::ObjReader       reader;
	tinyobj::ObjReaderConfig config;
	config.triangulate = true;
	if (reader.ParseFromFile(filename, config)) {
		m_meshes.clear();
		auto& shapes = reader.GetShapes();
		
		m_meshes.reserve(shapes.size());
		auto& attrib = reader.GetAttrib();
		
		for (auto& shape : shapes) {
			auto& shape_name = shape.name;
			auto& mesh       = shape.mesh;

			uint32_t idx_offset =  0;

			std::unordered_map<TinyobjIndexWrapper, uint32_t> index_map = {};
			std::unordered_set<uint16_t> mat_idx_set(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));
			auto mat_indices = std::vector<uint8_t>();
			auto materials   = std::vector<uint16_t>(std::begin(mat_idx_set), std::end(mat_idx_set));

			std::unordered_set<uint16_t> smoothing_group_idx_set(std::begin(shape.mesh.smoothing_group_ids), std::end(shape.mesh.smoothing_group_ids));
			auto smoothing_group_indices = std::vector<uint8_t>();
			auto smoothing_groups = std::vector<uint8_t>(std::begin(smoothing_group_idx_set), std::end(smoothing_group_idx_set));

			std::vector<ObjVec3> positions     = {};
			std::vector<ObjVec3> normals       = {};
			std::vector<ObjVec2> uvs           = {};
			std::vector<ObjIdx3> tri_indices   = {};
			uint32_t face_idx = 0;
			for (auto& num_face_vertex : mesh.num_face_vertices) {
				uint32_t idx3[3] = {};
				for (uint32_t f = 0; f < num_face_vertex; ++f) {
					auto& idx = mesh.indices[idx_offset + f];
					auto iter = index_map.find(TinyobjIndexWrapper(idx));
					if (iter != index_map.end()) {
						idx3[f] = iter->second;
					}
					else {
						auto v_idx = positions.size();
						auto vpx = attrib.GetVertices()[3u*idx.vertex_index+0u];
						auto vpy = attrib.GetVertices()[3u*idx.vertex_index+1u];
						auto vpz = attrib.GetVertices()[3u*idx.vertex_index+2u];
						positions.push_back({vpx,vpy,vpz});
						if (idx.normal_index > 0) {
							auto vnx = attrib.normals[3u * idx.normal_index + 0u];
							auto vny = attrib.normals[3u * idx.normal_index + 1u];
							auto vnz = attrib.normals[3u * idx.normal_index + 2u];
							normals.push_back({ vnx, vny,  vnz });
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
						idx3[f] = v_idx;
						index_map.insert({ TinyobjIndexWrapper(idx),v_idx });
					}
				}
				{

					auto mat_idx = shape.mesh.material_ids[face_idx];
					auto iter = std::find(std::begin(materials), std::end(materials), mat_idx);
					mat_indices.push_back(std::distance(std::begin(materials), iter));
				} 
				{

					auto smo_idx = shape.mesh.smoothing_group_ids[face_idx];
					auto iter = std::find(std::begin(smoothing_groups), std::end(smoothing_groups), smo_idx);
					smoothing_group_indices.push_back(std::distance(std::begin(smoothing_groups), iter));
				}
				
				tri_indices.push_back({ idx3[0],idx3[1],idx3[2] });
				idx_offset += num_face_vertex;
				face_idx++;
			}

			ObjBBox m_bbox;
			for (auto& position : positions) {
				m_bbox.addPoint(position);
			}
			
			m_meshes.insert({ 
				shape_name,
				{ std::move(tri_indices), std::move(positions), std::move(normals),std::move(uvs),m_bbox, mat_indices, materials, smoothing_group_indices, smoothing_groups}
			});
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
			auto mat         = ObjMaterial();
			mat.name         = material.name;
			mat.diffuse.x    = material.diffuse[0];
			mat.diffuse.y    = material.diffuse[1];
			mat.diffuse.z    = material.diffuse[2];
			mat.emission.x   = material.emission[0];
			mat.emission.y   = material.emission[1];
			mat.emission.z   = material.emission[2];
			mat.specular.x   = material.specular[0];
			mat.specular.y   = material.specular[1];
			mat.specular.z   = material.specular[2];
			mat.ior          = material.ior;
			mat.shinness     = material.shininess;
			mat.dissolve     = material.dissolve;
			mat.tex_diffuse  = get_tex_idx(material.diffuse_texname );
			mat.tex_specular = get_tex_idx(material.specular_texname);
			mat.tex_shinness = get_tex_idx(material.specular_highlight_texname);
			mat.tex_emission = get_tex_idx(material.emissive_texname);
			mat.tex_alpha    = get_tex_idx(material.alpha_texname);
			mat.tex_bump = get_tex_idx(material.bump_texname);
			mat.tex_reflection = get_tex_idx(material.reflection_texname);
			mat.tex_normal   = get_tex_idx(material.normal_texname);
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

auto hikari::test::owl::testlib::ObjMesh::getVisSmoothColors() const -> std::vector<ObjVec3>
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

auto hikari::test::owl::testlib::ObjMesh::getVisMaterialColors() const -> std::vector<ObjVec3>
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

auto hikari::test::owl::testlib::ObjMesh::getSubMeshIndices(uint32_t mat_idx) const -> std::vector<ObjIdx3>
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
