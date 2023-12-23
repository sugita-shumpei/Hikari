#include <hikari/texture/checkerboard.h>
#include <hikari/spectrum/uniform.h>
auto hikari::TextureCheckerboard::create()->std::shared_ptr<TextureCheckerboard> {
  return std::shared_ptr<TextureCheckerboard>(new hikari::TextureCheckerboard());
}
hikari::TextureCheckerboard::~TextureCheckerboard() {}
void hikari::TextureCheckerboard::setColor0(const SpectrumOrTexture& color0) { m_color0 = color0; }
void hikari::TextureCheckerboard::setColor1(const SpectrumOrTexture& color1) { m_color1 = color1; }
void hikari::TextureCheckerboard::setUVTransform(const Mat3x3& uv_transform) { m_uv_transform = uv_transform; }

auto hikari::TextureCheckerboard::getColor0()const->SpectrumOrTexture { return m_color0; }
auto hikari::TextureCheckerboard::getColor1()const->SpectrumOrTexture { return m_color1; }
auto hikari::TextureCheckerboard::getUVTransform() const->Mat3x3 { return m_uv_transform; }

hikari::Uuid hikari::TextureCheckerboard::getID() const { return ID(); }
hikari::TextureCheckerboard::TextureCheckerboard() :Texture(), m_color0{ SpectrumUniform::create(0.4) }, m_color1{ SpectrumUniform::create(0.2) }, m_uv_transform{Mat3x3(1.0f)} {}
