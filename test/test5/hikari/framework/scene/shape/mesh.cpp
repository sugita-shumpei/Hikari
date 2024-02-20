#include <hikari/shape/mesh.h>

auto hikari::shape::ShapeMeshObject::create(const Str& name) -> std::shared_ptr<ShapeMeshObject>
{
  return std::shared_ptr<ShapeMeshObject>(new ShapeMeshObject(name));
}

hikari::Str hikari::shape::ShapeMeshObject::getTypeString() const noexcept
{
    return TypeString();
}

bool hikari::shape::ShapeMeshObject::isConvertible(const Str& type) const noexcept
{
    return Convertible(type);
}

auto hikari::shape::ShapeMeshObject::getPropertyNames() const -> std::vector<Str>
{
  return {"name","indices","positions","normals","tangents","colors","uv","uv1","uv2","uv3","uv4","uv5","uv6","uv7","uv8","flip_normal","immutable"};
}

void hikari::shape::ShapeMeshObject::getPropertyBlock(PropertyBlock& pb) const
{
  pb.setValue("indices", getIndices());
}

void hikari::shape::ShapeMeshObject::setPropertyBlock(const PropertyBlock& pb)
{
}

bool hikari::shape::ShapeMeshObject::hasProperty(const Str& name) const {
    if (name == "name") { return true; }
    if (name == "indices") { return true; }
    if (name == "positions") { return true; }
    if (name == "normals") { return true; }
    if (name == "tangents") { return true; }
    if (name == "colors") { return true; }
    if (name == "uv") { return true; }
    if (name == "uv0") { return true; }
    if (name == "uv1") { return true; }
    if (name == "uv2") { return true; }
    if (name == "uv3") { return true; }
    if (name == "uv4") { return true; }
    if (name == "uv5") { return true; }
    if (name == "uv6") { return true; }
    if (name == "uv7") { return true; }
    if (name == "uv8") { return true; }
    if (name == "vertex_count") { return true; }
    if (name == "index_count") { return true; }
    if (name == "flip_normal") { return true; }
    if (name == "immutable") { return true; }
    return false;
}

bool hikari::shape::ShapeMeshObject::getProperty(const Str& name, PropertyBase<Object>& prop) const {
  if (name == "name") { prop.setValue(getName()); return true; }
  if (name == "flip_normals") {
    prop.setValue(getFlipNormals()); return true;
  }
  if (name == "vertex_count") { prop.setValue(getVertexCount()); return true; }
  if (name == "index_count") { prop.setValue(getIndexCount()); return true; }
  if (name == "index_format") { prop.setValue(convertEnum2Str(getIndexFormat())); return true; }
  if (name == "indices") { return impl_getIndices(prop); }
  if (name == "positions") { return impl_getPositions(prop); }
  if (name == "normals") { return impl_getNormals(prop); }
  if (name == "tangents") { return impl_getTangents(prop); }
  if (name == "colors") { return impl_getColors(prop); }
  if (name == "uv" ) { return  impl_getUV_without_idx_check(0, prop); }
  if (name == "uv0") { return  impl_getUV_without_idx_check(0, prop); }
  if (name == "uv1") { return  impl_getUV_without_idx_check(1, prop); }
  if (name == "uv2") { return  impl_getUV_without_idx_check(2, prop); }
  if (name == "uv3") { return  impl_getUV_without_idx_check(3, prop); }
  if (name == "uv4") { return  impl_getUV_without_idx_check(4, prop); }
  if (name == "uv5") { return  impl_getUV_without_idx_check(5, prop); }
  if (name == "uv6") { return  impl_getUV_without_idx_check(6, prop); }
  if (name == "uv7") { return  impl_getUV_without_idx_check(7, prop); }
  if (name == "uv8") { return  impl_getUV_without_idx_check(8, prop); }
  if (name == "immutable" ) { prop.setValue(isImmutable()); }
  return false;
}

bool hikari::shape::ShapeMeshObject::setProperty(const Str& name, const PropertyBase<Object>& prop) {
    if (name == "name") { auto str = prop.getValue<Str>(); if (str) { setName(*str); return true; } return false; }
    if (name == "index_format"){
      if (m_immutable) { return false; }
      auto str = prop.getValue<Str>();
      if (str) {
        auto tmp = convertStr2Enum<MeshIndexFormat>(*str);
        if (tmp) { setIndexFormat(*tmp); return true; }
      }
      return false;
    }
    if (name == "flip_normals") {
      if (m_immutable) { return false; }
      auto bv = prop.getValue<Bool>();
      if (bv) {
        setFlipNormals(*bv); return true;
      }
      return false;
    }
    if (name == "indices"  ) { return impl_setIndices(prop); }
    if (name == "positions") { return impl_setPositions(prop); }
    if (name == "normals"  ) { return impl_setNormals(prop); }
    if (name == "tangents" ) { return impl_setTangents(prop); }
    if (name == "colors"   ) { return impl_setColors(prop); }
    if (name == "uv")  { return  impl_setUV_without_idx_check(0, prop); }
    if (name == "uv0") { return  impl_setUV_without_idx_check(0, prop); }
    if (name == "uv1") { return  impl_setUV_without_idx_check(1, prop); }
    if (name == "uv2") { return  impl_setUV_without_idx_check(2, prop); }
    if (name == "uv3") { return  impl_setUV_without_idx_check(3, prop); }
    if (name == "uv4") { return  impl_setUV_without_idx_check(4, prop); }
    if (name == "uv5") { return  impl_setUV_without_idx_check(5, prop); }
    if (name == "uv6") { return  impl_setUV_without_idx_check(6, prop); }
    if (name == "uv7") { return  impl_setUV_without_idx_check(7, prop); }
    if (name == "uv8") { return  impl_setUV_without_idx_check(8, prop); }
    if (name == "immutable") {
      auto bv = prop.getValue<Bool>();
      if (bv) { if (*bv) { toImmutable(); return true; } }
    }
    return false;
}

auto hikari::shape::ShapeMeshObject::getBBox() const -> BBox3
{
    return m_bbox;
}

void hikari::shape::ShapeMeshObject::recalculateBBox()
{
  BBox3 new_bbox = {};
  if (m_index_count == 0) {
    for (auto& p : m_positions) {
      new_bbox.addPoint(p);
    }
  }
  else {
    for (auto& i : m_indices) {
      new_bbox.addPoint(m_positions[i]);
    }
  }
  m_bbox = new_bbox;
}

bool hikari::shape::ShapeMeshObject::impl_getIndices(Property& prop) const
{
  prop.setValue(getIndices());
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getPositions(Property& prop) const
{
  prop.setValue(getPositions());
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getNormals(Property& prop) const
{
  prop.setValue(getNormals());
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getTangents(Property& prop) const
{
  prop.setValue(getTangents());
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getColors(Property& prop) const
{
  prop.setValue(transformArray(getColors(), [](const auto& c) { return Vec4(c.r, c.g, c.b, c.a); }));
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getUV(Property& prop) const
{
  return impl_getUV_without_idx_check(0, prop);
}

bool hikari::shape::ShapeMeshObject::impl_getUV_without_idx_check(U32 idx, Property& prop) const
{
  prop.setValue(m_uvs[idx]);
  return true;
}

bool hikari::shape::ShapeMeshObject::impl_getUV(U32 idx, Property& prop) const
{
  if (idx >= 9) { return false; }
  return impl_getUV_without_idx_check(idx,prop);
}

bool hikari::shape::ShapeMeshObject::impl_setIndices(const Property& prop)
{
  if (!prop) { setIndices(Array<U32>()); return true; }
  auto array_u16 = prop.getValue<Array<U16>>();
  if (!array_u16.empty()) { setIndexFormat(MeshIndexFormat::eU16); setIndices(transformArray(array_u16, [](const auto& v) { return static_cast<U32>(v); })); return true; }
  auto array_u32 = prop.getValue<Array<U32>>();
  if (!array_u32.empty()) { setIndexFormat(MeshIndexFormat::eU32); setIndices(array_u32);return true;  }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setPositions(const Property& prop)
{
  if (!prop) { setPositions(Array<Vec3>()); return true; }
  auto array_vec2 = prop.getValue<Array<Vec2>>();
  if (!array_vec2.empty()) { setPositions(transformArray(array_vec2, [](const auto& v) { return static_cast<Vec3>(v,0.0f); }));return true;  }
  auto array_vec3 = prop.getValue<Array<Vec3>>();
  if (!array_vec3.empty()) { setPositions(array_vec3);return true;  }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setNormals(const Property& prop)
{
  if (!prop) { setNormals(Array<Vec3>()); return true; }
  auto array_vec3 = prop.getValue<Array<Vec3>>();
  if (!array_vec3.empty()) { setNormals(array_vec3); return true; }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setTangents(const Property& prop)
{
  if (!prop) { setTangents(Array<Vec4>()); return true; }
  auto array_vec4 = prop.getValue<Array<Vec4>>();
  if (!array_vec4.empty()) { setTangents(array_vec4); return true; }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setColors(const Property& prop)
{
  if (!prop) { setColors(Array<ColorRGBA>()); return true; }
  auto array_vec3 = prop.getValue<Array<Vec3>>();
  if (!array_vec3.empty()) {
    setColors(transformArray(array_vec3, [](const auto& v) { return ColorRGBA{ v.r, v.g, v.b,1.0f }; })); return true;
  }
  auto array_vec4 = prop.getValue<Array<Vec4>>();
  if (!array_vec4.empty()) {
    setColors(transformArray(array_vec4, [](const auto& v) { return ColorRGBA{ v.r, v.g, v.b, v.a }; })); return true;
  }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setUV(const Property& prop)
{
  return impl_setUV_without_idx_check(0, prop);
}

bool hikari::shape::ShapeMeshObject::impl_setUV_without_idx_check(U32 idx, const Property& prop)
{
  if (!prop) { setUV(idx, Array<Vec3>()); return true; }
  auto array_vec2 = prop.getValue<Array<Vec2>>();
  if (!array_vec2.empty()) {
    setUV(idx, array_vec2);
    return true;
  }
  auto array_vec3 = prop.getValue<Array<Vec3>>();
  if (!array_vec3.empty()) { setUV(idx, array_vec3); return true; }
  return false;
}

bool hikari::shape::ShapeMeshObject::impl_setUV(U32 idx, const Property& prop)
{
  if (idx >= 9) { return false; }
  return impl_setUV_without_idx_check(idx,prop);
}

auto hikari::shape::ShapeMeshDeserializer::getTypeString() const noexcept -> Str
{
  return ShapeMeshObject::TypeString();
}

auto hikari::shape::ShapeMeshDeserializer::eval(const Json& json) const -> std::shared_ptr<Object>
{
  auto iter_properties = json.find("properties");
  if (iter_properties == json.end()) { return nullptr; }
  auto& properties     = iter_properties.value();
  auto iter_name = json.find("name");
  if (iter_name == json.end()) { return nullptr; }
  if (!iter_name.value().is_string()) { return nullptr; }
  auto& iter_vertices = properties.find("vertices");
  if (iter_vertices == properties.end()) { return nullptr; }
  if (!iter_vertices.value().is_object()) { return nullptr; }
  auto vertices       = iter_vertices.value();
  auto name = iter_name.value().get<Str>();
  auto iter_positions = vertices.find("positions");
  if (iter_positions == vertices.end()) { return nullptr; }
  auto mesh = ShapeMesh(name);
  auto positions = iter_positions.value();
  do{
    auto positions_2d = deserialize<Array<Vec2>>(positions);
    if (!positions_2d.empty()) { mesh.setPositions(transformArray(positions_2d, [](const auto& v) { return Vec3(v, 0.0f); })); break; }
    auto positions_3d = deserialize<Array<Vec3>>(positions);
    if (!positions_3d.empty()) { mesh.setPositions(positions_3d);break;  }
    return nullptr;
  } while (false);
  auto iter_normals  = vertices.find("normals");
  if (iter_normals  != vertices.end()) {
    auto& normals = iter_normals.value();
    do {
      auto normals_3d = deserialize<Array<Vec3>>(normals);
      if (!normals_3d.empty()) { mesh.setNormals(normals_3d); break; }
      return nullptr;
    } while (false);
  }
  auto iter_tangents = vertices.find("tangents");
  if (iter_tangents != vertices.end()) {
    auto& tangents = iter_tangents.value();
    do {
      auto tangents_4d = deserialize<Array<Vec4>>(tangents);
      if (!tangents_4d.empty()) { mesh.setTangents(tangents_4d); break; }
      return nullptr;
    } while (false);
  }
  auto iter_colors   = vertices.find("colors");
  if (iter_colors   != vertices.end()) {
    auto& colors = iter_colors.value();
    do {
      auto colors_3d = deserialize<Array<Vec3>>(colors);
      if (!colors_3d.empty()) { mesh.setColors(transformArray(colors_3d, [](const auto& v) { return ColorRGBA{ v.r,v.g,v.b,1.0f }; })); break; }
      auto colors_4d = deserialize<Array<Vec4>>(colors);
      if (!colors_4d.empty()) { mesh.setColors(transformArray(colors_4d, [](const auto& v) { return ColorRGBA{ v.r,v.g,v.b,v.a }; })); break; }
      return nullptr;
    } while (false);
  }
  auto iter_uvs      = vertices.find("uvs");
  if (iter_uvs      != vertices.end()) {
    auto& uvs = iter_uvs.value();
    if (uvs.is_array()) {
      auto uv_arr = uvs.get<Array<Json>>();
      for (size_t i = 0; i < std::min(9ull, uv_arr.size()); ++i) {
        auto uv_2d = deserialize<Array<Vec2>>(uv_arr[i]);
        if (!uv_2d.empty()) { mesh.setUV(i, uv_2d); continue; }
        auto uv_3d = deserialize<Array<Vec3>>(uv_arr[i]);
        if (!uv_3d.empty()) { mesh.setUV(i, uv_3d); continue;  }
      }
    }
  }
  auto iter_indices = properties.find("indices");
  if (iter_indices != properties.end()) {
    auto& indices = iter_indices.value();
    if (indices.is_object()) {
      auto& iter_format = indices.find("format");
      if (iter_format != indices.end()) {
        if (iter_format.value().is_string()) {
          auto format = convertStr2Enum<MeshIndexFormat>(iter_format.value().get<Str>());
          if (format) {
            mesh.setIndexFormat(*format);
          }
        }
      }
      auto& iter_data = indices.find("data");
      if (iter_data != indices.end()) {
        auto data = deserialize<Array<U32>>(iter_data.value());
        mesh.setIndices(data);
      }
    }
  }
  auto iter_flip_normals = properties.find("flip_normals");
  if (iter_flip_normals != properties.end()) {
    if (iter_flip_normals.value().is_boolean()) {
      mesh.setFlipNormals(iter_flip_normals.value().get<bool>());
    }
  }
  return mesh.getObject();
}

auto hikari::shape::ShapeMeshSerializer::getTypeString() const noexcept -> Str
{
  return ShapeMeshObject::TypeString();
}

auto hikari::shape::ShapeMeshSerializer::eval(const std::shared_ptr<Object>& object) const -> Json
{
  auto shape_mesh = ObjectUtils::convert<ShapeMeshObject>(object);
  if (!shape_mesh) { return Json(); }
  Json json;
  json["type"] = "ShapeMesh";
  json["name"] = shape_mesh->getName();
  json["properties"] = {};
  json["properties"]["vertices"]  = {};
  json["properties"]["vertices"]["positions"] = serialize(shape_mesh->getPositions());
  if (shape_mesh->hasTangents()) json["properties"]["vertices"]["tangents" ] = serialize(shape_mesh->getTangents());
  if (shape_mesh->hasNormals()) json["properties"]["vertices"]["normals"  ] = serialize(shape_mesh->getNormals());
  if (shape_mesh->hasColors()) json["properties"]["vertices"]["colors"   ] = serialize(transformArray(shape_mesh->getColors(),[](const auto& c) { return Vec4(c.r, c.g, c.b, c.a); }));
  bool has_uv = shape_mesh->hasUV();
  if (has_uv){
    std::vector<Json> uvs(9);
    for (size_t i = 0; i < 9; ++i) {
      uvs[i] = serialize(shape_mesh->getUV(i));
    }
    json["properties"]["vertices"]["uvs"] = uvs;
  }
  if (shape_mesh->hasIndices()) {
    json["properties"]["indices"] = {};
    json["properties"]["indices"]["format"] = convertEnum2Str(shape_mesh->getIndexFormat());
    json["properties"]["indices"]["data"]   = serialize(shape_mesh->getIndices());
  }
  json["properties"]["bbox"] = {};
  auto bbox = shape_mesh->getBBox();
  json["properties"]["bbox"]["min"] = serialize(bbox.getMin());
  json["properties"]["bbox"]["max"]  = serialize(bbox.getMax());
  json["properties"]["flip_normals"] = shape_mesh->getFlipNormals();
  return json;
}
