#include "context.h"
#include <filesystem>
#include <sstream>
auto hikari::assets::mitsuba::XMLContext::create(const std::string& path, int major, int minor, int patch) noexcept -> std::shared_ptr<XMLContext>
{
  return std::shared_ptr<XMLContext>(new XMLContext(path,major,minor,patch));
}

hikari::assets::mitsuba::XMLContext::~XMLContext() noexcept {}

auto hikari::assets::mitsuba::XMLContext::clone() -> std::shared_ptr<XMLContext>
{
  auto res           = std::shared_ptr<XMLContext>(new XMLContext(m_path,m_version_major,m_version_minor,m_version_patch));
  res->m_alias_refs  = m_alias_refs ;
  res->m_def_values  = m_def_values ;
  res->m_ref_objects = m_ref_objects;
  res->m_sub_paths   = m_sub_paths  ;
  return res;
}

auto hikari::assets::mitsuba::XMLContext::normalizeRef(const std::string& id) const -> std::string
{
  auto norm_id = getAliasRef(id);
  if (norm_id == "") { return id; }
  return norm_id;
}

auto hikari::assets::mitsuba::XMLContext::normalizePaths(const std::string& relative_path) const -> std::vector<std::string>
{
  auto rel_path = std::filesystem::path(relative_path).lexically_normal();
  if (rel_path.is_absolute()) { return { rel_path.string()}; }

  auto res = std::vector<std::string>();
  for (auto& path : m_sub_paths) {
    res.push_back(std::filesystem::canonical(std::filesystem::path(m_path)/ path / rel_path).string());
  }
  return res;
}

auto hikari::assets::mitsuba::XMLContext::normalizeString(const std::string& input_str) const -> std::string
{
  std::vector<std::string> strs = {};
  std::stringstream ss; ss << input_str;
  std::string sentence;
  bool is_top_dollar = false;
  if (!input_str.empty()) {
    is_top_dollar = input_str[0] == '$';
  }
  while (std::getline(ss, sentence, '$')) {
    strs.push_back(sentence);
  }

  auto replaceTop = [this](const std::string& str) -> std::string {
    auto res = str;
    for (auto& [key, value] : m_def_values) {
      size_t pos = res.find(key);
      if (pos == std::string::npos) { continue; }
      size_t len = key.length();
      res.replace(pos, len, value);
    }
    return res;
  };
  bool is_first   = true;
  std::string res = "";
  for (auto& str : strs) {
    if (!is_first || is_top_dollar) {
      res += replaceTop(str);
    }
    else {
      res += str;
    }
    is_first = true;
  }
  return res;
}

bool hikari::assets::mitsuba::XMLContext::hasRef(const std::string& id) const
{
  if (hasRefObject(id)) { return true; }
  if (hasAliasRef (id)) { return true; }
  auto parent = m_parent_context.lock();
  if (parent) { return parent->hasRef(id); }
  return false;
}

auto hikari::assets::mitsuba::XMLContext::getObject(const std::string& id) const -> std::shared_ptr<XMLReferableObject>
{
  auto object = getRefObject(normalizeRef(id));
  if (object) { return object; }
  auto parent = m_parent_context.lock();
  if (parent) { return parent->getObject(id); }
  return nullptr;
}


void hikari::assets::mitsuba::XMLContext::setParentContext(const std::shared_ptr<XMLContext>& parent)
{
  auto parent_old = m_parent_context.lock();
  if (parent_old) {
    if (!parent) {
      auto iter = std::find(parent_old->m_child_contexts.begin(), parent_old->m_child_contexts.end(), shared_from_this());
      if (iter != parent_old->m_child_contexts.end()) { parent_old->m_child_contexts.erase(iter); }
      m_parent_context = {};
    }
    if (parent_old != parent) {
      if (parent) {
        m_parent_context = parent;
        parent->m_child_contexts.push_back(shared_from_this());
      }
    }
  }
  else {
    if (parent_old != parent) {
      m_parent_context = parent;
      parent->m_child_contexts.push_back(shared_from_this());
    }
  }
}

void hikari::assets::mitsuba::XMLContext::setPath(const std::string& path) {
  m_path =  (std::filesystem::path(path).lexically_normal()).string(); }


auto hikari::assets::mitsuba::XMLContext::getPath() const -> std::string { return m_path; }

void hikari::assets::mitsuba::XMLContext::addSubPath(const std::string& path) { auto path_ = std::filesystem::path(path).lexically_normal();  m_sub_paths.insert(path_.string()); }

void hikari::assets::mitsuba::XMLContext::popSubPath(const std::string& path) { m_sub_paths.erase(path); }

auto hikari::assets::mitsuba::XMLContext::getSubPathes() const -> std::vector<std::string> { return std::vector<std::string>(m_sub_paths.begin(), m_sub_paths.end()); }

void hikari::assets::mitsuba::XMLContext::setDefValue(const std::string& key, const std::string& value)
{
  m_def_values.insert({ key,value });
  auto parent = m_parent_context.lock();
  if (parent) {
    parent->setDefValue(key, value);
  }
}

auto hikari::assets::mitsuba::XMLContext::getDefValue(const std::string& key) const -> std::string
{
  auto iter = m_def_values.find(key);
  if (iter != m_def_values.end()) { return iter->second; }
  return "";
}

bool hikari::assets::mitsuba::XMLContext::hasDefValue(const std::string& key) const
{
  return m_def_values.count(key) > 0;
}

void hikari::assets::mitsuba::XMLContext::setAliasRef(const std::string& alias_id, const std::string& base_id)
{
  m_alias_refs.insert({ alias_id,base_id });
  auto parent = m_parent_context.lock();
  if (parent) {
    parent->setAliasRef(alias_id, base_id);
  }
}

auto hikari::assets::mitsuba::XMLContext::getAliasRef(const std::string& alias_id) const -> std::string
{
  auto iter = m_alias_refs.find(alias_id);
  if (iter != m_alias_refs.end()) { return iter->second; }
  return "";
}

bool hikari::assets::mitsuba::XMLContext::hasAliasRef(const std::string& alias_id) const
{
  return m_alias_refs.count(alias_id)>0;
}

bool hikari::assets::mitsuba::XMLContext::hasRefObject(const std::string& id) const
{
  return m_ref_objects.count(id)>0;
}

void hikari::assets::mitsuba::XMLContext::popRefObject(const std::string& id)
{
  m_ref_objects.erase(id);
  auto parent = m_parent_context.lock();
  if (parent) {
    m_ref_objects.erase(id);
  }
}

void hikari::assets::mitsuba::XMLContext::setRefObject(const std::shared_ptr<XMLReferableObject>& object)
{
  if (!object) { return; }
  auto id = object->getID();
  if (id == "") { return; }
  m_ref_objects.insert({id,object });
  auto parent = m_parent_context.lock();
  if (parent) {
    parent->setRefObject(object);
  }
}

auto hikari::assets::mitsuba::XMLContext::getRefObject(const std::string& id) const -> std::shared_ptr<XMLReferableObject>
{
  auto iter = m_ref_objects.find(id);
  if (iter != m_ref_objects.end()) { return iter->second; }
  else { return nullptr; };
}

auto hikari::assets::mitsuba::XMLContext::getRefObjects() const -> std::vector<std::shared_ptr<XMLReferableObject>>
{
  auto res = std::vector<std::shared_ptr<XMLReferableObject>>();
  for (auto& [ref, obj] : m_ref_objects) {
    res.push_back(obj);
  }
  return res;
}

auto hikari::assets::mitsuba::XMLContext::getRefShapes() const -> std::vector<std::shared_ptr<XMLReferableObject>>
{
  auto res = std::vector<std::shared_ptr<XMLReferableObject>>();
  for (auto& [ref, obj] : m_ref_objects) {
    if (obj->getObjectType() == XMLObjectType::eShape){
      res.push_back(obj);
    }
  }
  return res;
}

auto hikari::assets::mitsuba::XMLContext::getRefBsdfs() const -> std::vector<std::shared_ptr<XMLReferableObject>>
{
  auto res = std::vector<std::shared_ptr<XMLReferableObject>>();
  for (auto& [ref, obj] : m_ref_objects) {
    if (obj->getObjectType() == XMLObjectType::eBsdf) {
      res.push_back(obj);
    }
  }
  return res;
}

auto hikari::assets::mitsuba::XMLContext::getRefMediums() const -> std::vector<std::shared_ptr<XMLReferableObject>>
{
  auto res = std::vector<std::shared_ptr<XMLReferableObject>>();
  for (auto& [ref, obj] : m_ref_objects) {
    if (obj->getObjectType() == XMLObjectType::eMedium) {
      res.push_back(obj);
    }
  }
  return res;
}

auto hikari::assets::mitsuba::XMLContext::getRefTextures() const -> std::vector<std::shared_ptr<XMLReferableObject>>
{
  auto res = std::vector<std::shared_ptr<XMLReferableObject>>();
  for (auto& [ref, obj] : m_ref_objects) {
    if (obj->getObjectType() == XMLObjectType::eTexture) {
      res.push_back(obj);
    }
  }
  return res;
}

hikari::assets::mitsuba::XMLContext::XMLContext(const std::string& path, int major, int minor, int patch) noexcept
  :m_path{path},m_version_major{major},m_version_minor{minor},m_version_patch{patch}
{}
