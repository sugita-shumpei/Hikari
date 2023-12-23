#include <hikari/texture/bitmap.h>
auto hikari::TextureMipmap::create() -> std::shared_ptr<TextureMipmap> {
  return std::shared_ptr<TextureMipmap>(new TextureMipmap());
}
hikari::TextureMipmap::~TextureMipmap() {}
void hikari::TextureMipmap::setFilename(const String& filename) { m_filename = filename; }
void hikari::TextureMipmap::setFilterType(TextureFilterType filter_type) { m_filter_type = filter_type; }
void hikari::TextureMipmap::setWrapMode(TextureWrapMode wrapMode) { m_wrap_mode = wrapMode; }
void hikari::TextureMipmap::setRaw(Bool raw) { m_raw = raw; }
void hikari::TextureMipmap::setUVTransform(const Mat3x3& uv_transform) { m_uv_transform = uv_transform; }

auto hikari::TextureMipmap::getFilename() const->String { return m_filename; }
auto hikari::TextureMipmap::getFilterType() const->TextureFilterType { return m_filter_type; }
auto hikari::TextureMipmap::getWrapMode() const->TextureWrapMode { return m_wrap_mode; }
hikari::Bool hikari::TextureMipmap::getRaw() const { return m_rgb; }
auto hikari::TextureMipmap::getUVTransform() const->Mat3x3 { return m_uv_transform; }

hikari::TextureMipmap::TextureMipmap():Texture(),
m_filename{""},
m_filter_type{TextureFilterType::eBilinear},
m_wrap_mode{TextureWrapMode::eRepeat},
m_raw{false},
m_uv_transform{Mat3x3(1.0f)}
{
}

hikari::Uuid hikari::TextureMipmap::getID() const
{
  return ID();
}
