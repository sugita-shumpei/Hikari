#include <hikari/core/object.h>
#include <sstream>
auto hikari::core::Object::toString() const -> std::string
{
  auto to_indent = [](const std::string& in, const std::string indent)->std::string {
    std::string res;
    std::string tmp;
    std::stringstream ss; ss << in;
    while (std::getline(ss, tmp)) {
      res += indent + tmp;
      if (ss.rdbuf()->in_avail()) { res += "\n"; }
    }
    return res;
  };

  std::string res = "";
  res += "{\n";
  res += "  \"name\":\""      + m_name + "\"";
  res += ",\n" + to_indent("\"transform\":" + m_local_transform.toString() , "  ");
  if (!m_children.empty()) {
  res += ",\n  \"children\":[\n";
    auto i = 0;
    for (auto& child : m_children) {
      std::string end_str = (i == m_children.size() - 1) ? "\n" : ",\n";
      res += to_indent(child->toString(), "    ") + end_str;
      ++i;
    }
  res += "  ]\n";
  }
  else {
  res += "\n";
  }
  res += "}";
  return res;
}
