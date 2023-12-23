#pragma once
#include <memory>
#include <hikari/core/data_type.h>
#include <hikari/core/texture.h>
#include <hikari/core/mipmap.h>
namespace hikari {
  struct TextureMipmap : public Texture {
  public:
    static constexpr Uuid ID() { return Uuid::from_string("7B1250EA-8695-47E8-832D-7F6094D4D367").value(); }
    static auto create() -> std::shared_ptr<TextureMipmap>;
    virtual ~TextureMipmap();
    void setFilename(const String& filename);
    void setFilterType(TextureFilterType filter_type);
    void setWrapMode(TextureWrapMode wrapMode);
    void setRaw(Bool raw);
    void setUVTransform(const Mat3x3& uv_transform);
    void setMipmap(const MipmapPtr& mipmap);

    auto getFilename()   const->String;
    auto getFilterType() const->TextureFilterType;
    auto getWrapMode() const->TextureWrapMode;
    Bool getRaw() const;
    auto getUVTransform() const->Mat3x3;
    auto getMipmap() -> MipmapPtr;

    Uuid getID() const override;
  private:
    TextureMipmap();
  private:
    MipmapPtr         m_mipmap;
    String            m_filename;
    TextureFilterType m_filter_type;
    TextureWrapMode   m_wrap_mode;
    Bool              m_raw;
    Mat3x3            m_uv_transform;
  };
}
