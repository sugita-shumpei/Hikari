#pragma once
#include <hikari/assets/mitsuba/shape/mesh_data.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      namespace shape {
        struct PlyMeshImporterImpl : public MeshImporterImpl {
          PlyMeshImporterImpl(const std::string& filename) noexcept :m_filename{ filename } {}
          virtual ~PlyMeshImporterImpl() noexcept {}
          auto load() -> MeshImportOutput override;
        private:
          std::string m_filename;
        };
      }
    }
  }
}
