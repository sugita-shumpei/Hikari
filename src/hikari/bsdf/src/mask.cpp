#include <hikari/bsdf/mask.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfMask::create() -> std::shared_ptr<BsdfMask>
{
  return std::shared_ptr<BsdfMask>(new BsdfMask());
}

hikari::BsdfMask::~BsdfMask()
{
}

auto hikari::BsdfMask::getBsdf() -> BsdfPtr
{
  return m_bsdf;
}

void hikari::BsdfMask::setBsdf(const BsdfPtr& bsdf)
{
  m_bsdf = bsdf;
}

auto hikari::BsdfMask::getOpacity() -> SpectrumOrTexture
{
  return m_opacity;
}

void hikari::BsdfMask::setOpacity(const SpectrumOrTexture& texture)
{
  m_opacity = texture;
}

hikari::Uuid hikari::BsdfMask::getID() const
{
  return ID();
}

hikari::BsdfMask::BsdfMask()
  :Bsdf(),m_opacity{SpectrumUniform::create(0.5f)}, m_bsdf{}
{
}
