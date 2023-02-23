#ifndef HK_CORE_COMMON_JSON_H
#define HK_CORE_COMMON_JSON_H
#include <nlohmann/json.hpp>
using  HKJSON = nlohmann::json;
namespace HK
{
	using JSON = HKJSON;
	using namespace nlohmann;
}
#endif
