#pragma once
#include <hikari/core/data_type.h>
#include <unordered_map>
#include <optional>
namespace hikari {
  inline auto splitString(const String& str, Char delim) -> std::vector<String>
  {
    std::vector<String> res = {};
    std::size_t off = 0;
    std::size_t pos = {};
    while (off < str.size()) {
      pos = str.find_first_of(delim, off);
      if (pos == std::string::npos) {
        auto new_str = str.substr(off);
        res.push_back(new_str);
        return res;
      }
      else {
        auto new_str = str.substr(off, pos - off);
        res.push_back(new_str);
        off = pos + 1;
      }
    };
    return res;
  }
  template<typename Key, typename Val>
  inline auto getValueFromMap(const std::unordered_map<Key,Val>& map,const Key& key, const Val& def_val) -> Val
  {
    auto iter = map.find(key);
    if (iter == map.end()) { return def_val; }
    else { return iter->second; }
  }
  template<typename Key, typename Val>
  inline auto getValueFromMap(const std::unordered_map<Key, Val>& map, const Key& key) -> std::optional<Val>
  {
    auto iter = map.find(key);
    if (iter == map.end()) { return std::nullopt; }
    else { return iter->second; }
  }
}
