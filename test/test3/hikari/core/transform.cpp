#include <hikari/core/transform.h>
#include <nlohmann/json.hpp>

auto hikari::core::convertStringToTransform(const Str& v) -> Option<Transform>
{
  nlohmann::json j = nlohmann::json::parse(v);
  auto iter_matrix = j.find("matrix");
  if (iter_matrix != j.end()) {
    auto str_matrix = iter_matrix.value().dump();
    auto mat2 = convertStringToMat2(str_matrix); if (mat2) { return Transform(Mat4(*mat2)); }
    auto mat3 = convertStringToMat3(str_matrix); if (mat3) { return Transform(Mat4(*mat3)); }
    auto mat4 = convertStringToMat4(str_matrix); if (mat4) { return Transform(Mat4(*mat4)); }
    return std::nullopt;
  }
  TransformTRSData trs;
  auto iter_position = j.find("position");
  if (iter_position != j.end()) {
    auto str_position = iter_position.value().dump();
    auto pos1 = convertStringToF32 (str_position);
    auto pos2 = convertStringToVec2(str_position);
    auto pos3 = convertStringToVec3(str_position);
    if (pos1) { trs.position = hikari::core::Vec3(*pos1); }
    if (pos2) { trs.position = hikari::core::Vec3(*pos2,0.0f); }
    if (pos3) { trs.position = hikari::core::Vec3(*pos3); }
  }
  auto iter_rotation = j.find("rotation");
  if (iter_rotation != j.end()) {
    auto str_rotation = iter_rotation.value().dump();
    auto rotation = convertStringToQuat(str_rotation);
    if (rotation) { trs.rotation = *rotation; }
  }
  auto iter_scale = j.find("scale");
  if (iter_scale != j.end()) {
    auto str_scale = iter_scale.value().dump();
    auto scl1 = convertStringToF32(str_scale);
    auto scl2 = convertStringToVec2(str_scale);
    auto scl3 = convertStringToVec3(str_scale);
    if (scl1) { trs.scale = hikari::core::Vec3(*scl1); }
    if (scl2) { trs.scale = hikari::core::Vec3(*scl2, 0.0f); }
    if (scl3) { trs.scale = hikari::core::Vec3(*scl3); }
  }
  return Transform(trs);
}

auto hikari::core::convertJSONStringToTransform(const Str& v) -> Option<Transform>
{
  auto j = nlohmann::json::parse(v);
  auto iter_type = j.find("type");
  if (iter_type == j.end()) { return std::nullopt; }
  auto& type = iter_type.value();
  if (!type.is_string()) { return std::nullopt; }
  if (type.get<std::string>() != "Transform") { return std::nullopt; }
  return convertStringToTransform(j.dump());
}
