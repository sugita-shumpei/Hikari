#pragma once
#include <hikari/core/data_type.h>
#include <zlib.h>
#include <fstream>
#include <sstream>
#include <vector>

namespace hikari {
  // Data
  enum       MitsubaSerializedDataFlags : U32 {
    MitsubaSerializedDataFlagsHasVertexNormal = 0x0001,
    MitsubaSerializedDataFlagsHasVertexUV     = 0x0002,
    MitsubaSerializedDataFlagsHasVertexColor  = 0x0008,
    MitsubaSerializedDataFlagsUseFaceNormal   = 0x0010,
    MitsubaSerializedDataFlagsSinglePrecision = 0x1000,
    MitsubaSerializedDataFlagsDoublePrecision = 0x2000,
  };
  struct     MitsubaSerializedMeshData {
    ~MitsubaSerializedMeshData() noexcept {}
    void load(const Byte* data, U64 size);

    std::vector<F64>  m_vertex_positions = {};
    std::vector<F64>  m_vertex_normals   = {};
    std::vector<F64>  m_vertex_uvs       = {};
    std::vector<F64>  m_vertex_colors    = {};
    std::vector<U64>  m_faces            = {};
    U32               m_flags            = 0;
    U64               m_vertex_count     = 0;
    U64               m_face_count       = 0;
    String            m_name             = "";
    Bool              m_loaded           = false;
  };
  enum class MitsubaSerializedLoadType {
    eFull,
    eTemp
  };
  struct     MitsubaSerializedData {
      MitsubaSerializedData() noexcept {}
     ~MitsubaSerializedData() noexcept {}

    Bool load(const String&  filename, MitsubaSerializedLoadType load_type = MitsubaSerializedLoadType::eFull);

    bool loadMesh(U32 idx);
    auto getMeshes() const -> const std::vector<MitsubaSerializedMeshData>& { return m_meshes; }

    U16  getFormat() const;
    U16  getVersion()const;

    auto getBinaryData() const -> const Byte*;
    auto getBinarySize() const -> U64;
    void releaseBinary();
  private:
    std::vector<MitsubaSerializedMeshData> m_meshes            = {};
    std::vector<U64>                       m_offsets           = {};
    std::vector<U64>                       m_sizes             = {};
    std::unique_ptr<Byte[]>                m_binary_data       = nullptr;
    U64                                    m_binary_size       = 0;
    String                                 m_filename          = "";
    U16                                    m_format            = 0;
    U16                                    m_version           = 0;
    Bool                                   m_loaded            = false;
    Bool                                   m_has_binary        = false;
  };
}
