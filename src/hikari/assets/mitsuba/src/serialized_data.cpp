#include "serialized_data.h"


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
      if (!m_has_binary) { return m_meshes[idx].m_loaded; }
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

    ss.read((Char*)&m_vertex_count, sizeof(m_vertex_count));
    ss.read((Char*)&m_face_count, sizeof(m_face_count));

    bool isSinglePrecision = (m_flags & MitsubaSerializedDataFlagsSinglePrecision);

    auto readFloats = [&ss, &isSinglePrecision](U64 count) {
      if (isSinglePrecision) {
        auto temp_data = std::vector<F32>(count);
        ss.read((Char*)temp_data.data(), sizeof(F32) * temp_data.size());
        auto f64_data = std::vector<F64>(count);
        std::ranges::copy(temp_data, f64_data.begin());
        return f64_data;
      }
      else {
        auto temp_data = std::vector<F64>(count);
        ss.read((Char*)temp_data.data(), sizeof(F64) * temp_data.size());
        return temp_data;
      }
      };
    auto readUInts = [&ss, &isSinglePrecision](U64 count) {
      if (isSinglePrecision) {
        auto temp_data = std::vector<U32>(count);
        ss.read((Char*)temp_data.data(), sizeof(U32) * temp_data.size());
        auto f64_data = std::vector<U64>(count);
        std::ranges::copy(temp_data, f64_data.begin());
        return f64_data;
      }
      else {
        auto temp_data = std::vector<U64>(count);
        ss.read((Char*)temp_data.data(), sizeof(U64) * temp_data.size());
        return temp_data;
      }
      };
    {
      m_vertex_positions = readFloats(3 * m_vertex_count);
    }
    if (m_flags & MitsubaSerializedDataFlagsHasVertexNormal) {
      m_vertex_normals = readFloats(3 * m_vertex_count);
    }
    if (m_flags & MitsubaSerializedDataFlagsHasVertexUV) {
      m_vertex_uvs = readFloats(2 * m_vertex_count);
    }
    if (m_flags & MitsubaSerializedDataFlagsHasVertexColor) {
      m_vertex_colors = readFloats(3 * m_vertex_count);
    }
    m_faces = readUInts(3 * m_face_count);
  }

  m_loaded = true;
}
