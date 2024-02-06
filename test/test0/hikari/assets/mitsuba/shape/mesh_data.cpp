#include <hikari/assets/mitsuba/shape/mesh_data.h>
#include <hikari/assets/mitsuba/shape/serialized/mesh_importer.h>
#include <hikari/assets/mitsuba/shape/obj/mesh_importer.h>
#include <hikari/assets/mitsuba/shape/ply/mesh_importer.h>
#include <filesystem>

auto hikari::assets::mitsuba::shape::Mesh::create(const std::string& name, const std::vector<uint32_t>& faces, const std::vector<float>& vertex_positions, const std::vector<float>& vertex_normals, const std::vector<float>& vertex_texcoords, const std::unordered_map<std::string, std::vector<float>>& mesh_attributes, bool use_face_normals) -> std::shared_ptr<Mesh>
{
  auto res= std::shared_ptr<Mesh>(new Mesh());
  res->m_name             = name;
  res->m_face_count       = faces.size() / 3;
  res->m_vertex_count     = vertex_positions.size() / 3;
  res->m_faces            = faces;
  res->m_vertex_positions = vertex_positions;
  res->m_vertex_normals   = vertex_normals;
  res->m_vertex_texcoords = vertex_texcoords;
  res->m_mesh_attributes  = mesh_attributes;
  res->m_use_face_normal = use_face_normals;
  return res;
}

auto hikari::assets::mitsuba::shape::MeshImporter::create(const std::string& filename) -> std::shared_ptr<MeshImporter>
{
  auto path = std::filesystem::path(filename);
  if (!std::filesystem::exists(path)) { return nullptr; }
  if (path.extension() == ".serialized") {
    auto res = std::shared_ptr<MeshImporter>(new MeshImporter());
    res->m_impl.reset(new SerializedMeshImporterImpl(filename));
    return res;
  }
  if (path.extension() == ".obj") {
    auto res = std::shared_ptr<MeshImporter>(new MeshImporter());
    res->m_impl.reset(new ObjMeshImporterImpl(filename));
    return res;
  }
  if (path.extension() == ".ply") {
    auto res = std::shared_ptr<MeshImporter>(new MeshImporter());
    res->m_impl.reset(new PlyMeshImporterImpl(filename));
    return res;
  }
  return std::shared_ptr<MeshImporter>();
}

hikari::assets::mitsuba::shape::MeshImporter::~MeshImporter() noexcept
{
}

auto hikari::assets::mitsuba::shape::MeshImporter::load() -> MeshImportOutput
{
  if (m_impl) {
    return m_impl->load();
  }
  return MeshImportOutput();
}

hikari::assets::mitsuba::shape::MeshImporter::MeshImporter() noexcept
  :m_impl{nullptr}
{
}
