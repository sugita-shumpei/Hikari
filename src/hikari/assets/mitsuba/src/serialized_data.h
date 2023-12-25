#pragma once
#include <hikari/core/data_type.h>
#include <hikari/shape/mesh.h>
#include <zlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

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
     MitsubaSerializedMeshData() noexcept {}
    ~MitsubaSerializedMeshData() noexcept {}
    MitsubaSerializedMeshData(const MitsubaSerializedMeshData&) noexcept = default;
    MitsubaSerializedMeshData& operator=(const MitsubaSerializedMeshData&) noexcept = default;
    void load(const Byte* data, U64 size);

    auto getData()              const -> const Byte* { return m_data.get(); }
    template<typename T>
    auto getVertexPositionPtr() const -> const T*    { return reinterpret_cast<const T*>(m_data.get()); }
    template<typename T>
    auto getVertexNormalPtr()   const -> const T*    { return hasVertexNormal()? reinterpret_cast<const T*>(m_data.get() + getVertexNormalOffset()): nullptr; }
    template<typename T>
    auto getVertexUVPtr()       const -> const T*    { return hasVertexUV()    ? reinterpret_cast<const T*>(m_data.get() + getVertexUVOffset())    : nullptr; }
    template<typename T>
    auto getVertexColorPtr()    const -> const T*    { return hasVertexColor() ? reinterpret_cast<const T*>(m_data.get() + getVertexColorOffset()) : nullptr; }
    template<typename T>
    auto getFacePtr()           const -> const T*    { return reinterpret_cast<const T*>(m_data.get() + getFaceOffset()); }

    auto getTypeSizeInBytes()    const -> U64 { return (m_flags & MitsubaSerializedDataFlagsSinglePrecision) ? 4 : 8; }
    auto getVertexNormalOffset() const -> U64 { return                           (hasVertexNormal() ? m_vertex_count * 3 * getTypeSizeInBytes() : 0); }
    auto getVertexUVOffset()     const -> U64 { return getVertexNormalOffset() + (hasVertexUV()     ? m_vertex_count * 2 * getTypeSizeInBytes() : 0); }
    auto getVertexColorOffset()  const -> U64 { return getVertexUVOffset()     + (hasVertexColor()  ? m_vertex_count * 3 * getTypeSizeInBytes() : 0); }
    auto getFaceOffset()         const -> U64 { return getVertexColorOffset()  + m_face_count * 3 * getTypeSizeInBytes(); }

    bool hasVertexNormal() const { return (m_flags & MitsubaSerializedDataFlagsHasVertexNormal); }
    bool hasVertexColor()  const { return (m_flags & MitsubaSerializedDataFlagsHasVertexColor ); }
    bool hasVertexUV()     const { return (m_flags & MitsubaSerializedDataFlagsHasVertexUV    ); }
    bool getFaceNormal()   const { return (m_flags & MitsubaSerializedDataFlagsUseFaceNormal  ); }

    auto getFlags()       const -> U32           { return m_flags;        }
    auto getVertexCount() const -> U64           { return m_vertex_count; }
    auto getFaceCount()   const -> U64           { return m_face_count;   }
    auto getName()        const -> const String& { return m_name;         }
    Bool isLoaded()       const                  { return m_loaded;       }
  private:
    std::shared_ptr<Byte[]> m_data           = nullptr;
    U32                     m_flags          = 0;
    U64                     m_vertex_count   = 0;
    U64                     m_face_count     = 0;
    U64                     m_data_size      = 0;
    String                  m_name           = "";
    Bool                    m_loaded         = false;
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
    auto getMeshCount()const->U32 { return m_meshes.size(); }
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

  struct     ShapeMeshMitsubaSerialized : public ShapeMesh
  {
    static auto create(const MitsubaSerializedMeshData& data) -> std::shared_ptr<ShapeMeshMitsubaSerialized>;
    virtual ~ShapeMeshMitsubaSerialized();
    auto getVertexCount() const->U32 override;
    auto getFaceCount() const->U32 override;
    void clear() override;
    auto getVertexPositions() const->std::vector<Vec3> override;
    auto getVertexNormals() const->std::vector<Vec3> override;
    auto getVertexBinormals() const->std::vector<Vec4> override;
    auto getVertexUVs() const->std::vector<Vec2> override;
    auto getVertexColors() const->std::vector<Vec3> override;
    auto getFaces() const->std::vector<U32> override;
    void setVertexPositions(const std::vector<Vec3>& vertex_positions) override;
    void setVertexNormals(const std::vector<Vec3>& vertex_normals) override;
    void setVertexBinormals(const std::vector<Vec4>& vertex_binormals) override;
    void setVertexUVs(const std::vector<Vec2>& vertex_uvs) override;
    void setVertexColors(const std::vector<Vec3>& vertex_colors) override;
    void setFaces(const std::vector<U32>& faces) override;
    Bool getFlipUVs() const override;
    Bool getFaceNormals() const override;
    void setFlipUVs(Bool flip_uvs) override;
    void setFaceNormals(Bool face_normals) override;
    Bool hasVertexNormals() const override;
    Bool hasVertexBinormals() const override;
    Bool hasVertexUVs() const override;
    Bool hasVertexColors() const override;
  private:
    ShapeMeshMitsubaSerialized(const MitsubaSerializedMeshData& data);
  private:
    static auto convertToArrayVec2(const void* p_data, U32 count, U32 size) -> std::vector<Vec2> {
      if (!p_data) { return {}; }
      std::vector<Vec2> res(count);
      if (size == 4)
      { for (size_t i = 0; i < size; ++i) { res[i] = {((const F32*)p_data)[2*i+0],((const F32*)p_data)[2 * i + 1] }; } }
      else 
      { for (size_t i = 0; i < size; ++i) { res[i] = {((const F64*)p_data)[2*i+0],((const F64*)p_data)[2 * i + 1] }; } }
      return res;
    }
    static auto convertToArrayVec3(const void* p_data, U32 count, U32 size) -> std::vector<Vec3> {
      if (!p_data) { return {}; }
      std::vector<Vec3> res(count);
      if (size == 4)
      {
        for (size_t i = 0; i < size; ++i) { res[i] = { ((const F32*)p_data)[3 * i + 0],((const F32*)p_data)[3 * i + 1],((const F32*)p_data)[3 * i + 2] }; }
      }
      else
      {
        for (size_t i = 0; i < size; ++i) { res[i] = { ((const F64*)p_data)[3 * i + 0],((const F64*)p_data)[3 * i + 1],((const F64*)p_data)[3 * i + 2] }; }
      }
      return res;
    }
    static auto convertToArrayU32 (const void* p_data, U32 count, U32 size) -> std::vector<U32> {
      if (!p_data) { return {}; }
      std::vector<U32> res(count);
      if (size == 4)
      {
        for (size_t i = 0; i < size; ++i) { res[i] = ((const U32*)p_data)[i]; }
      }
      else
      {
        for (size_t i = 0; i < size; ++i) { res[i] = ((const U64*)p_data)[i]; }
      }
      return res;
    }
  private:
    MitsubaSerializedMeshData m_data;
    bool m_face_normals;
    bool m_flip_uvs;
  };

}
