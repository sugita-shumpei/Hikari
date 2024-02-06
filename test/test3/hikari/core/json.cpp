#include <hikari/core/json.h>
#include <unordered_set>
auto hikari::core::convertStringToJSON(const Str& str) -> Json
{
  // 単純にJSONをパースする
  return nlohmann::json::parse(str);
}

auto hikari::core::convertJSONToString(const Json& json) -> Str
{
  // 単純にJSONをダンプする
  return nlohmann::to_string(json);
}
