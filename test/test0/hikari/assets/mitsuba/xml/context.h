#pragma once
#include <unordered_set>
#include "object.h"
namespace hikari {
  namespace assets {
    namespace mitsuba {
      struct XMLSceneImporter;
      struct XMLReferableObject;
      struct XMLBsdf;
      struct XMLTexture;
      struct XMLMedium;
      struct XMLShape;
      struct XMLContext : public std::enable_shared_from_this<XMLContext> {
        static auto create(const std::string& path, int major, int minor, int patch) noexcept -> std::shared_ptr<XMLContext>;
        ~XMLContext() noexcept;

        auto clone() -> std::shared_ptr<XMLContext>; 

        void getVersion(int& major, int& minor, int& patch)const { major = m_version_major; minor = m_version_minor; patch = m_version_patch; }
        auto getVersionString() const -> std::string { return std::to_string(m_version_major) + "." + std::to_string(m_version_minor) + "." + std::to_string(m_version_patch); }

        auto normalizeRef(const std::string& id) const->std::string;
        auto normalizePaths(const std::string& relative_path) const->std::vector<std::string>;
        auto normalizeString(const std::string& str) const -> std::string;

        bool hasRef(const std::string& id) const;
        auto getObject(const std::string& id) const->std::shared_ptr<XMLReferableObject>;

        auto getPath() const -> std::string;

        auto getSubPathes() const -> std::vector<std::string>;

        auto getDefValue(const std::string& key) const-> std::string;
        bool hasDefValue(const std::string& key) const;

        auto getAliasRefs() const->std::vector<std::pair<std::string,std::string>> { return std::vector<std::pair<std::string, std::string>>(m_alias_refs.begin(),m_alias_refs.end()); }
        auto getAliasRef(const std::string& alias_id) const->std::string;
        bool hasAliasRef(const std::string& alias_id) const;

        bool hasRefObject(const std::string& id) const;
        auto getRefObject(const std::string& id) const -> std::shared_ptr<XMLReferableObject>;

        auto getRefObjects () const->std::vector<std::shared_ptr<XMLReferableObject>>;
        auto getRefShapes  () const->std::vector<std::shared_ptr<XMLReferableObject>>;
        auto getRefBsdfs   () const->std::vector<std::shared_ptr<XMLReferableObject>>;
        auto getRefMediums () const->std::vector<std::shared_ptr<XMLReferableObject>>;
        auto getRefTextures() const->std::vector<std::shared_ptr<XMLReferableObject>>;

        auto getParentContext() const -> std::shared_ptr<XMLContext>              { return m_parent_context.lock(); }
        auto getChildContexts() const -> std::vector<std::shared_ptr<XMLContext>> { return m_child_contexts; }

      private:
        friend class XMLSceneImporter;
        friend class XMLBsdf;
        friend class XMLTexture;
        friend class XMLMedium;
        friend class XMLShape;
        void setParentContext(const std::shared_ptr<XMLContext>& parent);
        void setPath(const std::string& path);
        void addSubPath(const std::string& path);
        void popSubPath(const std::string& path);
        void setDefValue(const std::string& key, const std::string& value);
        void setAliasRef(const std::string& alias_id, const std::string& base_id);
        void setRefObject(const std::shared_ptr<XMLReferableObject>& object);
        void popRefObject(const std::string& id);
      private:
        XMLContext(const std::string& path, int major, int minor, int patch) noexcept;
        int                                                                  m_version_major  = 0;
        int                                                                  m_version_minor  = 0;
        int                                                                  m_version_patch  = 0;
        std::string                                                          m_path           = "";
        std::unordered_set<std::string>                                      m_sub_paths      = {};
        std::unordered_map<std::string, std::string>                         m_alias_refs     = {};
        std::unordered_map<std::string, std::shared_ptr<XMLReferableObject>> m_ref_objects    = {};
        std::unordered_map<std::string, std::string>                         m_def_values     = {};
        std::weak_ptr<XMLContext>                                            m_parent_context = {};
        std::vector<std::shared_ptr<XMLContext>>                             m_child_contexts = {};
      };
      using XMLContextPtr = std::shared_ptr<XMLContext>;
    }
  }
}
