#pragma once
#include <memory>
#include <hikari/core/variant.h>
#include <hikari/core/data_type.h>
#include <hikari/core/texture.h>
#include <hikari/core/mipmap.h>
namespace hikari {
  struct TextureCheckerboard : public Texture {
  public:
    static constexpr Uuid ID() { return Uuid::from_string("957EE9D8-575A-447D-8B90-3BAFF012AF5F").value(); }
    static auto create() -> std::shared_ptr<TextureCheckerboard>;
    virtual ~TextureCheckerboard();
    void setColor0(const SpectrumOrTexture& spec_or_text0);
    void setColor1(const SpectrumOrTexture& spec_or_text1);
    void setUVTransform(const Mat3x3& uv_transform);

    auto getColor0()const-> SpectrumOrTexture;
    auto getColor1()const-> SpectrumOrTexture;
    auto getUVTransform() const->Mat3x3;

    Uuid getID() const override;
  private:
    TextureCheckerboard();
  private:
    SpectrumOrTexture m_color0;
    SpectrumOrTexture m_color1;
    Mat3x3            m_uv_transform;
  };
}
