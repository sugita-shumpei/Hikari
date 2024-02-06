#pragma once
#include "../mesh_data.h"
#include <variant>
#include <optional>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      namespace shape {
        struct SerializedMesh {
          enum   Flags : uint32_t {
            FlagsVertexNormal    = 0x0001,
            FlagsVertexTexcoord  = 0x0002,
            FlagsVertexColor     = 0x0008,
            FlagsFaceNormal      = 0x0010,
            FlagsSinglePrecision = 0x1000,
            FlagsDoublePrecision = 0x2000
          };
          struct DescData {
            std::string          name;
            uint64_t     num_vertices;
            uint64_t        num_faces;
          };
          struct SinglePrecisionArrayData {
            std::vector<float>    vertex_positions;
            std::vector<float>    vertex_normals;
            std::vector<float>    vertex_texcoords;
            std::vector<float>    vertex_colors;
          };
          struct DoublePrecisionArrayData{
            auto toSingle() const -> SinglePrecisionArrayData {
              SinglePrecisionArrayData res = {};
              res.vertex_positions = std::vector<float>(vertex_positions.begin(), vertex_positions.end());
              res.vertex_normals   = std::vector<float>(  vertex_normals.begin(),   vertex_normals.end());
              res.vertex_texcoords = std::vector<float>(vertex_texcoords.begin(), vertex_texcoords.end());
              res.vertex_colors    = std::vector<float>(   vertex_colors.begin(),    vertex_colors.end());
              return res;
            }
            std::vector<double>   vertex_positions;
            std::vector<double>   vertex_normals;
            std::vector<double>   vertex_texcoords;
            std::vector<double>   vertex_colors;
          };
          struct CompressedData {
            auto toMesh() const->std::shared_ptr<Mesh>;
            uint32_t                                flags = 0x0000;
            DescData                                 desc = {};
            std::variant<
              SinglePrecisionArrayData,
              DoublePrecisionArrayData
            >     vertex_array = SinglePrecisionArrayData();
            std::variant<
              std::vector<uint32_t>,
              std::vector<uint64_t>
            >                                        faces = std::vector<uint32_t>{};
          };

          uint16_t                                  format = 0;
          uint16_t                                 version = 0;
          CompressedData                        compressed = {};
        };
        struct SerializedHeader {
          std::vector<uint64_t>      offsets;
          uint32_t                num_meshes;
        };
        struct SerializedData {
          auto toMeshes() const->std::vector<std::shared_ptr<Mesh>>;
          uint64_t                    size_in_bytes;
          std::vector<SerializedMesh> meshes;
          SerializedHeader            header;
        };
        struct SerializedDataImporter {
           SerializedDataImporter() noexcept {}
          ~SerializedDataImporter() noexcept {}
          auto load(const std::string& filename) -> std::optional<SerializedData>;
        };
        struct SerializedMeshImporterImpl : public MeshImporterImpl {
          SerializedMeshImporterImpl(const std::string& filename) noexcept :m_filename{ filename } {}
          virtual ~SerializedMeshImporterImpl() noexcept {}
          auto load() -> MeshImportOutput override;
        private:
          std::string m_filename;
        };
      }
    }
  }
}
