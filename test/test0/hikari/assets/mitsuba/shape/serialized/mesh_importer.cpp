#include "mesh_importer.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <zlib.h>

#define HK_ASSETS_MITSUBA_SERIALIZED_FILEFORMAT    0x041C
#define HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_3 0x0003
#define HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_4 0x0004

auto hikari::assets::mitsuba::shape::SerializedMesh::CompressedData::toMesh() const -> std::shared_ptr<Mesh>
{
  using namespace std::string_literals;
  std::string name   = desc.name;
  if (faces.index() != 0) { return nullptr; }
  if (vertex_array.index() == 0) {
    auto& array_data = std::get<0>(vertex_array);

    return Mesh::create(
      name,
      std::get<0>(faces),
      array_data.vertex_positions,
      array_data.vertex_normals,
      array_data.vertex_texcoords,
      array_data.vertex_colors.empty() ? std::unordered_map<std::string,std::vector<float>>{} : std::unordered_map<std::string, std::vector<float>>{ { "vertex_colors"s, array_data.vertex_colors } },
      flags&FlagsFaceNormal
    );
  }
  else {
    auto array_data = std::get<1>(vertex_array).toSingle();
    return Mesh::create(
      name,
      std::get<0>(faces),
      array_data.vertex_positions,
      array_data.vertex_normals,
      array_data.vertex_texcoords,
      array_data.vertex_colors.empty() ? std::unordered_map<std::string, std::vector<float>>{} : std::unordered_map<std::string, std::vector<float>>{ { "vertex_colors"s, array_data.vertex_colors } },
      flags & FlagsFaceNormal
    );
  }
}

auto hikari::assets::mitsuba::shape::SerializedData::toMeshes() const -> std::vector<std::shared_ptr<Mesh>>
{
  auto res= std::vector<std::shared_ptr<Mesh>>();
  res.reserve(header.num_meshes);
  for (auto& mesh : meshes) {
    res.push_back(mesh.compressed.toMesh());
  }
  return res;
}

auto hikari::assets::mitsuba::shape::SerializedDataImporter::load(const std::string& filename) -> std::optional<SerializedData>
{
  std::ifstream       file(filename, std::ios::binary);
  if (file.fail()) { return std::nullopt; }
  std::vector<char> byte_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  auto file_size  = byte_data.size();
  auto format     = uint16_t(0);
  auto version    = uint16_t(0);
  auto num_meshes = uint32_t(0);

  size_t min_size = sizeof(uint32_t) + 2*sizeof(uint16_t);
  if (file_size < min_size) { return std::nullopt;  }
  else {
    std::memcpy(&format    , &byte_data[0]                           , sizeof(uint16_t));
    std::memcpy(&version   , &byte_data[sizeof(uint16_t)]            , sizeof(uint16_t));
    std::memcpy(&num_meshes, &byte_data[file_size - sizeof(uint32_t)], sizeof(uint32_t));
  }

  if (format  != HK_ASSETS_MITSUBA_SERIALIZED_FILEFORMAT) { return std::nullopt; }
  if ((version!= HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_3)&&
      (version!= HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_4)){
    return std::nullopt;
  }

  SerializedHeader header =   {};
  header.num_meshes       = num_meshes;

  size_t file_end = file_size - sizeof(uint32_t);
  if (version == HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_3) {
    file_end -= sizeof(uint32_t) * num_meshes;
    min_size += sizeof(uint32_t) * num_meshes;
    if (file_size < min_size) { return std::nullopt; }
    auto tmp_offsets  = std::vector<uint32_t>(num_meshes,0);
    std::memcpy(tmp_offsets.data(), &byte_data[file_size - sizeof(uint32_t) - sizeof(uint32_t) * num_meshes], sizeof(uint32_t) * num_meshes);
    header.offsets = std::vector<uint64_t>(tmp_offsets.begin(), tmp_offsets.end());
  }
  else {
    file_end -= sizeof(uint64_t) * num_meshes;
    min_size += sizeof(uint64_t) * num_meshes;
    if (file_size < min_size) { return std::nullopt; }
    auto tmp_offsets = std::vector<uint64_t>(num_meshes, 0);
    std::memcpy(tmp_offsets.data(), &byte_data[file_size - sizeof(uint32_t) - sizeof(uint64_t) * num_meshes], sizeof(uint64_t) * num_meshes);
    header.offsets = tmp_offsets;
  }

  for (auto& offset : header.offsets) {
    uint16_t tmp_format  = 0u;
    uint16_t tmp_version = 0u;
    std::memcpy(&tmp_format , &byte_data[offset                  ], sizeof(uint16_t));
    std::memcpy(&tmp_version, &byte_data[offset+ sizeof(uint16_t)], sizeof(uint16_t));

    if (tmp_format   != HK_ASSETS_MITSUBA_SERIALIZED_FILEFORMAT) { return std::nullopt; }
    if ((tmp_version != HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_3) &&
       (tmp_version   != HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_4)) {
      return std::nullopt;
    }
  }

  uint8_t output_buffer[10240] = {};
  std::vector<SerializedMesh> meshes(num_meshes);
  for (auto i = 0; i < header.offsets.size();++i) {
    meshes[i].format  = format;
    meshes[i].version = version;
    size_t load_beg   = header.offsets[i] + sizeof(uint16_t) * 2;
    size_t load_end   = 0;
    if (i != header.offsets.size() - 1) {
      load_end = header.offsets[i + 1];
    }
    else {
      load_end = file_end;
    }

    if (load_beg >= load_end) { return std::nullopt; }
    size_t load_size = load_end - load_beg;
    {
      z_stream  zs;
      zs.zalloc = Z_NULL;
      zs.zfree  = Z_NULL;
      zs.opaque = Z_NULL;

      if (inflateInit(&zs) != Z_OK) { return std::nullopt; }

      zs.next_in  = (Bytef*)&byte_data[load_beg];
      zs.avail_in = load_end - load_beg;

      std::memset(output_buffer, 0, sizeof(output_buffer));
      int ret = 0;
      uint64_t    read_pos = 0;

      std::vector<uint8_t> out;
      do {
        zs.next_out  = (Bytef*)output_buffer;
        zs.avail_out = sizeof(output_buffer);
        ret          = inflate(&zs, 0);
        auto outsize = sizeof(output_buffer) - zs.avail_out;
        out.reserve(out.size() + outsize);
        std::copy(std::begin(output_buffer), std::next(output_buffer, outsize), std::back_inserter(out));
      } while (ret == Z_OK);

      inflateEnd(&zs);
      if (ret != Z_STREAM_END) { return std::nullopt; }

      size_t out_pos      = 0;
      size_t out_size     = out.size();
      size_t out_min_size = sizeof(uint32_t);
      if (out_size < out_min_size) { return std::nullopt; }

      uint32_t     flags = 0;
      std::memcpy(&flags, &out[out_pos], sizeof(uint32_t));
      out_pos += sizeof(uint32_t);

      meshes[i].compressed.flags = flags;

      std::string name = "";
      if (version == HK_ASSETS_MITSUBA_SERIALIZED_FILEVERSION_4) {
        auto len   = std::strlen((const char*)&out[out_pos]);
        name = std::string((const char* const)&out[out_pos], len);
        out_pos += (len + 1);
        out_min_size += (len + 1);
      }
      meshes[i].compressed.desc.name = name;

      out_min_size += (sizeof(uint64_t) * 2);
      if (out_size < out_min_size) { return std::nullopt; }
      uint64_t num_vertices = 0;
      uint64_t num_faces    = 0;
      {
        std::memcpy(&num_vertices, &out[out_pos], sizeof(num_vertices));
        out_pos += sizeof(uint64_t);
        std::memcpy(&num_faces, &out[out_pos], sizeof(num_faces));
        out_pos += sizeof(uint64_t);
      }
      meshes[i].compressed.desc.num_vertices = num_vertices;
      meshes[i].compressed.desc.num_faces    = num_faces;

      uint64_t vertex_data_size = 3 * num_vertices;
      {
        if (flags& SerializedMesh::FlagsVertexNormal) {
          vertex_data_size += 3 * num_vertices;
        }
        if (flags & SerializedMesh::FlagsVertexTexcoord) {
          vertex_data_size += 2 * num_vertices;
        }
        if (flags & SerializedMesh::FlagsVertexColor) {
          vertex_data_size += 3 * num_vertices;
        }
        if (flags & SerializedMesh::FlagsSinglePrecision) {
          vertex_data_size *= 4;
        }
        else {
          vertex_data_size *= 8;
        }
      }
      out_min_size += vertex_data_size;
      if (out_size < out_min_size) { return std::nullopt; }
      if (flags & SerializedMesh::FlagsSinglePrecision) {
        auto array_data = SerializedMesh::SinglePrecisionArrayData();
        array_data.vertex_positions.resize(3 * num_vertices);
        std::memcpy(array_data.vertex_positions.data(),  &out[out_pos], sizeof(float) * 3 * num_vertices)    ; out_pos     += sizeof(float) * 3 * num_vertices;
        if (flags & SerializedMesh::FlagsVertexNormal  ){ array_data.vertex_normals.resize(3 * num_vertices);std::memcpy(array_data.vertex_normals.data()        , &out[out_pos], sizeof(float) * 3 * num_vertices)  ; out_pos += sizeof(float) * 3 * num_vertices; }
        if (flags & SerializedMesh::FlagsVertexTexcoord){ array_data.vertex_texcoords.resize(2 * num_vertices); std::memcpy(array_data.vertex_texcoords.data()   , &out[out_pos], sizeof(float) * 2 * num_vertices)  ; out_pos += sizeof(float) * 2 * num_vertices;}
        if (flags & SerializedMesh::FlagsVertexColor   ){ array_data.vertex_colors.resize(3 * num_vertices); std::memcpy(array_data.vertex_colors.data()         , &out[out_pos], sizeof(float) * 3 * num_vertices)  ; out_pos += sizeof(float) * 3 * num_vertices;}
        meshes[i].compressed.vertex_array = std::move(array_data);
      }
      else {
        auto array_data = SerializedMesh::DoublePrecisionArrayData();
        array_data.vertex_positions.resize(3 * num_vertices);
        std::memcpy(array_data.vertex_positions.data(),  &out[out_pos], sizeof(double) * 3 * num_vertices)   ; out_pos     += sizeof(double) * 3 * num_vertices;
        if (flags & SerializedMesh::FlagsVertexNormal  ){ array_data.vertex_normals.resize(3 * num_vertices)  ; std::memcpy(array_data.vertex_normals.data()   , &out[out_pos], sizeof(double) * 3 * num_vertices)  ; out_pos += sizeof(double) * 3 * num_vertices;}
        if (flags & SerializedMesh::FlagsVertexTexcoord){ array_data.vertex_texcoords.resize(2 * num_vertices); std::memcpy(array_data.vertex_texcoords.data() , &out[out_pos], sizeof(double) * 2 * num_vertices)  ; out_pos += sizeof(double) * 2 * num_vertices;}
        if (flags & SerializedMesh::FlagsVertexColor   ){ array_data.vertex_colors.resize(3 * num_vertices)   ; std::memcpy(array_data.vertex_colors.data()    , &out[out_pos], sizeof(double) * 3 * num_vertices)  ; out_pos += sizeof(double) * 3 * num_vertices;}
        meshes[i].compressed.vertex_array = std::move(array_data);
      }

      uint32_t face_data_size = 3*num_faces;
      if (num_vertices >= 0xFFFFFFFFu) {
        face_data_size *= 8;
        out_min_size += face_data_size;
        if (out_size < out_min_size) { return std::nullopt; }
        auto faces = std::vector<uint64_t>(num_faces * 3);
        std::memcpy(faces.data(), &out[out_pos], sizeof(uint64_t) * 3 * num_faces);
        meshes[i].compressed.faces = std::move(faces);
      }
      else {
        face_data_size *= 4;
        out_min_size += face_data_size;
        if (out_size < out_min_size) { return std::nullopt; }
        auto faces = std::vector<uint32_t>(num_faces * 3);
        std::memcpy(faces.data(), &out[out_pos], sizeof(uint32_t) * 3 * num_faces);
        meshes[i].compressed.faces = std::move(faces);
      }

    }
  }
  SerializedData data = {};
  data.header = header;
  data.meshes = std::move(meshes);
  data.size_in_bytes = file_size;
  return data;
}

auto hikari::assets::mitsuba::shape::SerializedMeshImporterImpl::load() -> MeshImportOutput
{
  auto data_importer = SerializedDataImporter();
  auto data = data_importer.load(m_filename);
  if (!data) { return {}; }
  else { return { data->toMeshes(), nullptr }; }
}
