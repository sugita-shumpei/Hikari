#pragma once
#include "../mesh_data.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      namespace shape {
        struct ObjMeshImporterImpl : public MeshImporterImpl {
          ObjMeshImporterImpl(const std::string& filename) noexcept :m_filename{ filename } {}
          virtual ~ObjMeshImporterImpl() noexcept {}
          auto load() -> MeshImportOutput override;
        private:
          std::string m_filename;
        };
      }
    }
  }
}
