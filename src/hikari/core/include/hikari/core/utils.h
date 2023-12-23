#pragma once
#include <hikari/core/data_type.h>

namespace hikari {
  inline auto   splitString(const String& str, Char delim) -> std::vector<String>
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
}
