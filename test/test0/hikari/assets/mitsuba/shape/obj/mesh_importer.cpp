#include <hikari/assets/mitsuba/shape/obj/mesh_importer.h>
#include <tiny_obj_loader.h>
#include <unordered_set>

auto hikari::assets::mitsuba::shape::ObjMeshImporterImpl::load() -> MeshImportOutput
{
  auto obj_reader         = tinyobj::ObjReader();
  auto obj_config         = tinyobj::ObjReaderConfig();

  struct Idx3 {
    Idx3(const tinyobj::index_t& idx) :m_idx{ idx } {

    }
    bool operator==(const Idx3& idx)const {
      if (m_idx.normal_index != idx.m_idx.normal_index) { return false; }
      if (m_idx.texcoord_index != idx.m_idx.texcoord_index) { return false; }
      if (m_idx.vertex_index != idx.m_idx.vertex_index) { return false; }
      return true;
    }

    tinyobj::index_t m_idx;
  };
  struct Idx3Hash {
    size_t operator()(const Idx3& idx3) const noexcept {
      unsigned long long tmp1;
      int vals[] = { idx3.m_idx.vertex_index ,idx3.m_idx.normal_index };
      std::memcpy(&tmp1, vals, sizeof(tmp1));
      return tmp1;
    }
  };
  obj_config.triangulate     = true;
  obj_config.vertex_color    = true;
  // Parseを実行
  if (!obj_reader.ParseFromFile(m_filename,obj_config)) { return MeshImportOutput(); }
  // 
  auto& shapes            = obj_reader.GetShapes();
  auto& attrib            = obj_reader.GetAttrib();
  auto& materials         = obj_reader.GetMaterials();
  // 
  auto res                = MeshImportOutput();
  auto shape_materials    = std::vector<int>();
  shape_materials.reserve(shapes.size());
  for (auto& shape : shapes) {
    auto mat_datas = std::vector<std::tuple<
      std::vector<float>,
      std::vector<float>,
      std::vector<float>,
      std::vector<float>,
      std::vector<uint32_t>,
      std::string
    >>();
    auto mat_set   = std::unordered_set<int>(std::begin(shape.mesh.material_ids), std::end(shape.mesh.material_ids));
    auto hash_maps = std::vector<std::unordered_map<Idx3, size_t, Idx3Hash>>(mat_set.size());
    mat_datas.resize(mat_set.size());
    size_t num_faces = 0;
    for (auto& mat_idx : mat_set) {
      shape_materials.push_back(mat_idx);
    }
    for (auto& num_face : shape.mesh.num_face_vertices) { num_faces += num_face; }
    {
      size_t i = 0;
      for (auto& mat_data : mat_datas) {
        {
          std::get<0>(mat_data).reserve(3 * num_faces);
          std::get<1>(mat_data).reserve(3 * num_faces);
          std::get<2>(mat_data).reserve(2 * num_faces);
          std::get<3>(mat_data).reserve(3 * num_faces);
          std::get<4>(mat_data).reserve(    num_faces);
        }
        if (mat_set.size() >= 2) {
          std::get<5>(mat_data) = shape.name + "." + std::to_string(i);
        }
        else {
          std::get<5>(mat_data) = shape.name;
        }
        ++i;
      }
    }
    {
      size_t face_off = 0;
      for (size_t face_idx = 0; face_idx < shape.mesh.num_face_vertices.size(); ++face_idx) {
        auto  mat_idx  = shape.mesh.material_ids[face_idx];
        auto  idx      = std::distance(std::begin(mat_set), mat_set.find(mat_idx));
        auto& mat_data = mat_datas[idx];
        auto& hash_map = hash_maps[idx];
        for (size_t fv = 0; fv < shape.mesh.num_face_vertices[face_idx]; ++fv) {
          auto& idx3 = shape.mesh.indices[face_off + fv];
          auto  iter = hash_map.find(idx3);
          auto  mesh_idx = size_t(0);
          if (iter == std::end(hash_map)) {
            mesh_idx = hash_map.size();
            hash_map.insert({ idx3,mesh_idx });
            {
              std::get<0>(mat_data).push_back(attrib.GetVertices()[3 * idx3.vertex_index + 0]);
              std::get<0>(mat_data).push_back(attrib.GetVertices()[3 * idx3.vertex_index + 1]);
              std::get<0>(mat_data).push_back(attrib.GetVertices()[3 * idx3.vertex_index + 2]);
            }
            if (idx3.normal_index >= 0) {
              std::get<1>(mat_data).push_back(attrib.normals[3 * idx3.normal_index + 0]);
              std::get<1>(mat_data).push_back(attrib.normals[3 * idx3.normal_index + 1]);
              std::get<1>(mat_data).push_back(attrib.normals[3 * idx3.normal_index + 2]);
            }
            if (idx3.texcoord_index >= 0) {
              std::get<2>(mat_data).push_back(attrib.texcoords[2 * idx3.texcoord_index + 0]);
              std::get<2>(mat_data).push_back(attrib.texcoords[2 * idx3.texcoord_index + 1]);
            }
            if (!attrib.colors.empty()) {
              std::get<3>(mat_data).push_back(attrib.colors[3 * idx3.vertex_index + 0]);
              std::get<3>(mat_data).push_back(attrib.colors[3 * idx3.vertex_index + 1]);
              std::get<3>(mat_data).push_back(attrib.colors[3 * idx3.vertex_index + 2]);
            }
          }
          else {
            mesh_idx = iter->second;
          }
          std::get<4>(mat_data).push_back(mesh_idx);
        }
        face_off += shape.mesh.num_face_vertices[face_idx];
      }
    }
    for (auto& mat_data : mat_datas) {
      std::get<0>(mat_data).shrink_to_fit();
      std::get<1>(mat_data).shrink_to_fit();
      std::get<2>(mat_data).shrink_to_fit();
      std::get<3>(mat_data).shrink_to_fit();
      std::get<4>(mat_data).shrink_to_fit();

      res.meshes.push_back(
        Mesh::create(
          std::get<5>(mat_data),
          std::get<4>(mat_data),
          std::get<0>(mat_data),
          std::get<1>(mat_data),
          std::get<2>(mat_data),
          std::get<3>(mat_data).empty() ?
          std::unordered_map<std::string, std::vector<float>>{} :
          std::unordered_map<std::string, std::vector<float>>{ {"colors",std::get<3>(mat_data)}}
        )
      );
    }
  }

  return res;
}
