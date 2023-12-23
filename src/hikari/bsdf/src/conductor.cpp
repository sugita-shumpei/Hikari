#include <hikari/bsdf/conductor.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfConductor::create() -> std::shared_ptr<BsdfConductor>
{
  return std::shared_ptr<BsdfConductor>(new BsdfConductor());
}

hikari::BsdfConductor::~BsdfConductor()
{
}

hikari::Uuid hikari::BsdfConductor::getID() const { return ID(); }

void hikari::BsdfConductor::setEta(const SpectrumOrTexture& eta)
{
  m_eta = eta;
}

void hikari::BsdfConductor::setK(const SpectrumOrTexture& k)
{
  m_k = k;
}

auto hikari::BsdfConductor::getEta() const -> SpectrumOrTexture
{
  return m_eta;
}

auto hikari::BsdfConductor::getK() const -> SpectrumOrTexture
{
  return m_k;
}

auto hikari::BsdfConductor::getSpecularReflectance() const -> SpectrumOrTexture
{
  return m_specular_reflectance;
}

void hikari::BsdfConductor::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

hikari::BsdfConductor::BsdfConductor():Bsdf(),
m_eta{},
m_k{},
m_specular_reflectance{SpectrumUniform::create(1.0f)}
{
}
