
#include <hikari/core/variant.h>
#include <hikari/spectrum/uniform.h>
#include <hikari/spectrum/srgb.h>
#include <hikari/texture/checkerboard.h>
#include <hikari/texture/mipmap.h>

int main() {
  hikari::SpectrumOrTexture spec(hikari::SpectrumUniform::create(1.0f));

  auto checker = hikari::TextureCheckerboard::create();
  checker->setColor0(hikari::TextureMipmap  ::create());
  checker->setColor1(hikari::SpectrumUniform::create());


}
