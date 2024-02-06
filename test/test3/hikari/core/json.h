#pragma once
#include <nlohmann/json.hpp>
#include <hikari/core/data_type.h>
namespace hikari {
  inline namespace core {
    using Json = nlohmann::json;

    auto convertStringToJSON(const Str&   str)  -> Json;
    auto convertJSONToString(const Json& json)  -> Str;
  }
}
