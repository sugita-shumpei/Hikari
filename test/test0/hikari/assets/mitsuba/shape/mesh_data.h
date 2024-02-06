#pragma once
#include <hikari/assets/mitsuba/xml/bsdf.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      namespace shape {
        struct Mesh {
          using MeshAttributes = std::unordered_map<std::string, std::vector<float>>;
          static auto create(
            const std::string&           name,
            const std::vector<uint32_t>& faces,
            const std::vector<float>&    vertex_positions,
            const std::vector<float>&    vertex_normals,
            const std::vector<float>&    vertex_texcoords,
            const MeshAttributes&        mesh_attributes = {},
            bool use_face_normals = false
          ) -> std::shared_ptr<Mesh>;
          
          ~Mesh() noexcept {}

          auto getName() const -> std::string { return m_name; }

          auto getVertexCount()     const -> uint64_t { return m_vertex_count; }
          auto getFaceCount()       const -> uint64_t { return m_face_count; }
          auto getFaces()           const -> const std::vector<uint32_t>& { return m_faces; }
          auto getVertexPositions() const -> const std::vector<float>& { return m_vertex_positions; }
          auto getVertexNormals()   const -> const std::vector<float>& { return m_vertex_normals; }
          auto getVertexTexCoords() const -> const std::vector<float>& { return m_vertex_texcoords; }
          auto getMeshAttributes(const std::string& name) const -> const std::vector<float>* {
            auto iter = m_mesh_attributes.find(name);
            if (iter != m_mesh_attributes.end()) { return &iter->second; }
            else { return nullptr; }
          }

          bool hasVertexPositions() const { return m_vertex_positions.size() != 0; }
          bool hasVertexNormals()   const { return m_vertex_normals.size() != 0; }
          bool hasVertexTexCoords() const { return m_vertex_texcoords.size() != 0; }
          bool hasMeshAttributes(const std::string& name) const { return m_mesh_attributes.count(name) != 0; }
        private:
          Mesh() noexcept {}
        private:
          std::string                                         m_name             = "";
          uint64_t                                            m_vertex_count     = 0;
          uint64_t                                            m_face_count       = 0;
          std::vector<uint32_t>                               m_faces            = {};
          std::vector<float>                                  m_vertex_positions = {};
          std::vector<float>                                  m_vertex_normals   = {};
          std::vector<float>                                  m_vertex_texcoords = {};
          std::unordered_map<std::string, std::vector<float>> m_mesh_attributes  = {};
          bool                                                m_use_face_normal  = false;
        };
        struct MeshExtData {
          virtual ~MeshExtData() {}
        };
        struct MeshImportOutput {
          std::vector<std::shared_ptr<Mesh>> meshes = {};
          std::shared_ptr<MeshExtData>       ext    = {};
        };
        struct MeshImporterImpl {
          virtual ~MeshImporterImpl() {}
          virtual auto load() -> MeshImportOutput = 0;
        };
        struct MeshImporter {
          static auto create(const std::string& filename) ->std::shared_ptr<MeshImporter>;
          virtual ~MeshImporter() noexcept;
          auto load() -> MeshImportOutput ;
        private:
          MeshImporter() noexcept;
        private:
          std::unique_ptr<MeshImporterImpl> m_impl;
        };
      }
    }
  }
}
