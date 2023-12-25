#include "serialized_data.h"
#include <hikari/shape/mesh.h>


hikari::Bool hikari::MitsubaSerializedData::load(const String& filename, MitsubaSerializedLoadType load_type) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) { return false; }

    file.seekg(0L, std::ios::end);
    auto binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0L, std::ios::beg);
    m_format  = 0u;
    m_version = 0u;
    {
        auto read_offset = static_cast<U64>(0);
        // format
        file.seekg(read_offset, std::ios::beg);
        file.read((Char*)&m_format, sizeof(m_format));
        if (m_format != 0x041C) { return false; }
        // version
        file.read((Char*)&m_version, sizeof(m_version));
    }
    // load header
    {
        U32 mesh_count = 0u;
        file.seekg(binary_size-sizeof(U32), std::ios::beg);
        file.read((Char*)&mesh_count, sizeof(mesh_count));
        m_offsets.resize(mesh_count);
        m_sizes.resize(mesh_count);
        U64 read_offset = binary_size - (sizeof(U32) + sizeof(U64) * mesh_count);
        file.seekg(read_offset, std::ios::beg);
        for (size_t i = 0; i < mesh_count; ++i) {
            U64 offset = 0;
            file.read((Char*)&offset, sizeof(offset));
            m_offsets[i] = offset;
        }

        if (mesh_count > 1) {
          for (size_t i = 0; i < mesh_count-1; ++i) {
            m_sizes[i] = m_offsets[i + 1] - m_offsets[i];
          }
        }
        m_sizes[mesh_count - 1] = read_offset - m_offsets[mesh_count - 1];
    }
    m_filename = filename;
    // data_offsets and sizes
    {
        m_binary_data = std::unique_ptr<Byte[]>(new Byte[binary_size]);
        m_binary_size = binary_size;
        m_meshes.clear();
        file.seekg(0, std::ios::beg);
        file.read((Char*)m_binary_data.get(), binary_size);
    }

    m_meshes.resize(m_offsets.size());

    if (load_type == MitsubaSerializedLoadType::eTemp) {
        m_loaded     = true;
        m_has_binary = true;
        return true;
    }
    else {
        m_loaded      = true;
        m_has_binary  = true;
        for (U64 i = 0; i < m_meshes.size(); ++i) {
            loadMesh(i);
        }
        m_binary_data.reset();
        m_binary_size = 0;
        m_has_binary  = false;
        return true;
    }

    file.close();
}

bool hikari::MitsubaSerializedData::loadMesh(U32 idx) {
    if (!m_loaded    ) { return false; }
    if (idx < m_meshes.size()) {
      if (!m_has_binary) { return m_meshes[idx].isLoaded(); }
      auto header_size = sizeof(U16) * 2;
      m_meshes[idx].load(m_binary_data.get() + m_offsets[idx] + header_size, m_sizes[idx]- header_size);
      return true;
    }
    return false;
}

hikari::U16 hikari::MitsubaSerializedData::getFormat() const { return m_format; }

hikari::U16 hikari::MitsubaSerializedData::getVersion() const { return m_version; }

auto hikari::MitsubaSerializedData::getBinaryData() const -> const Byte* { return m_binary_data.get(); }

auto hikari::MitsubaSerializedData::getBinarySize() const -> U64 { return m_binary_size; }

void hikari::MitsubaSerializedData::releaseBinary() {
    m_has_binary = false;
    m_binary_data.reset();
    m_binary_size = 0;
}

void hikari::MitsubaSerializedMeshData::load(const Byte* data, U64 size) {
  if (m_loaded) { return; }

  z_stream zs;
  zs.zalloc = Z_NULL;
  zs.zfree = Z_NULL;
  zs.opaque = Z_NULL;

  if (inflateInit(&zs) != Z_OK) {
    return;
  }

  zs.next_in = (Bytef*)data;
  zs.avail_in = size;

  //出力バッファは10KBを用意する
  Char output_buffer[10240];
  int ret;
  U64         read_pos = 0;
  std::string outstring;
  do {
    zs.next_out = (Bytef*)output_buffer;
    zs.avail_out = sizeof(output_buffer);
    ret = inflate(&zs, 0);
    // 読み込めたら順番に処理を実行する
    if (outstring.size() < zs.total_out) {
      outstring.append(output_buffer, zs.total_out - outstring.size());
    }
  } while (ret == Z_OK);
  inflateEnd(&zs);
  if (ret != Z_STREAM_END) { return; }

  {
    std::stringstream ss(outstring);
    ss.read((Char*)&m_flags, sizeof(m_flags));
    String name = "";
    Char ch = '\0';
    do {
      ss.read(&ch, sizeof(ch));
      name.push_back(ch);
    } while (ch != '\0');
    m_name = name;
    // vertex, face情報を読み取る
    ss.read((Char*)&m_vertex_count, sizeof(m_vertex_count));
    ss.read((Char*)&m_face_count  , sizeof(m_face_count)  );
    // 精度情報を取得
    bool isSinglePrecision = (m_flags & MitsubaSerializedDataFlagsSinglePrecision);
    auto element_type_size = isSinglePrecision ? 4 : 8;

    auto whole_size = m_vertex_count * 3 * element_type_size + m_face_count * 3 * element_type_size;
    if (m_flags & MitsubaSerializedDataFlagsHasVertexNormal) {
      whole_size += m_vertex_count * 3 * element_type_size;
    }
    if (m_flags & MitsubaSerializedDataFlagsHasVertexUV) {
      whole_size += m_vertex_count * 2 * element_type_size;
    }
    if (m_flags & MitsubaSerializedDataFlagsHasVertexColor) {
      whole_size += m_vertex_count * 3 * element_type_size;
    }
    m_data      = std::unique_ptr<Byte[]>(new Byte[whole_size]);
    m_data_size = whole_size;

    ss.read((Char*)m_data.get(), whole_size);
  }

  m_loaded = true;
}

auto hikari::ShapeMeshMitsubaSerialized::create(const MitsubaSerializedMeshData& data) -> std::shared_ptr<ShapeMeshMitsubaSerialized>
{
  return std::shared_ptr<ShapeMeshMitsubaSerialized>(new ShapeMeshMitsubaSerialized(data));
}

hikari::ShapeMeshMitsubaSerialized::~ShapeMeshMitsubaSerialized()
{
}

auto hikari::ShapeMeshMitsubaSerialized::getVertexCount() const -> U32 { return m_data.getVertexCount(); }

auto hikari::ShapeMeshMitsubaSerialized::getFaceCount() const -> U32 { return m_data.getFaceCount(); }

void hikari::ShapeMeshMitsubaSerialized::clear() { return;}

auto hikari::ShapeMeshMitsubaSerialized::getVertexPositions() const -> std::vector<Vec3> {

  return convertToArrayVec3(m_data.getVertexPositionPtr<void>(),m_data.getVertexCount(),m_data.getTypeSizeInBytes());
}

auto hikari::ShapeMeshMitsubaSerialized::getVertexNormals() const -> std::vector<Vec3>
{
  return convertToArrayVec3(m_data.getVertexNormalPtr<void>(), m_data.getVertexCount(), m_data.getTypeSizeInBytes());
}

auto hikari::ShapeMeshMitsubaSerialized::getVertexBinormals() const -> std::vector<Vec4>
{
  return std::vector<Vec4>();
}

auto hikari::ShapeMeshMitsubaSerialized::getVertexUVs() const -> std::vector<Vec2>
{
  return convertToArrayVec2(m_data.getVertexUVPtr<void>(), m_data.getVertexCount(), m_data.getTypeSizeInBytes());
}

auto hikari::ShapeMeshMitsubaSerialized::getVertexColors() const -> std::vector<Vec3>
{
  return convertToArrayVec3(m_data.getVertexColorPtr<void>(), m_data.getVertexCount(), m_data.getTypeSizeInBytes());
}

auto hikari::ShapeMeshMitsubaSerialized::getFaces() const -> std::vector<U32>
{
  return convertToArrayU32(m_data.getFacePtr<void>(), m_data.getFaceCount()*3, m_data.getTypeSizeInBytes());
}

void hikari::ShapeMeshMitsubaSerialized::setVertexPositions(const std::vector<Vec3>& vertex_positions)
{
  return;
}

void hikari::ShapeMeshMitsubaSerialized::setVertexNormals(const std::vector<Vec3>& vertex_normals)
{
  return;
}

void hikari::ShapeMeshMitsubaSerialized::setVertexBinormals(const std::vector<Vec4>& vertex_binormals)
{
  return;
}

void hikari::ShapeMeshMitsubaSerialized::setVertexUVs(const std::vector<Vec2>& vertex_uvs)
{
  return;
}

void hikari::ShapeMeshMitsubaSerialized::setVertexColors(const std::vector<Vec3>& vertex_colors)
{
  return;
}

void hikari::ShapeMeshMitsubaSerialized::setFaces(const std::vector<U32>& faces)
{
  return;
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::getFlipUVs() const
{
  return m_flip_uvs;
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::getFaceNormals() const
{
  return m_face_normals;
}

void hikari::ShapeMeshMitsubaSerialized::setFlipUVs(Bool flip_uvs)
{
  m_flip_uvs = flip_uvs;
}

void hikari::ShapeMeshMitsubaSerialized::setFaceNormals(Bool face_normals)
{
  m_face_normals = face_normals;
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::hasVertexNormals() const
{
  return m_data.hasVertexNormal();
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::hasVertexBinormals() const
{
  return false;
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::hasVertexUVs() const
{
  return m_data.hasVertexUV();
}

hikari::Bool hikari::ShapeMeshMitsubaSerialized::hasVertexColors() const
{
  return m_data.hasVertexColor();
}

hikari::ShapeMeshMitsubaSerialized::ShapeMeshMitsubaSerialized(const MitsubaSerializedMeshData& data)
  :m_data{data},m_face_normals{ data.getFaceNormal()}, m_flip_uvs{false}
{
}
