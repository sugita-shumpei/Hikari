#include <hikari/bsdf/rough_conductor.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfRoughConductor::create() -> std::shared_ptr<BsdfRoughConductor>
{
  return std::shared_ptr<BsdfRoughConductor>(new BsdfRoughConductor());
}

hikari::BsdfRoughConductor::~BsdfRoughConductor()
{
}

hikari::Uuid hikari::BsdfRoughConductor::getID() const { return ID(); }

void hikari::BsdfRoughConductor::setEta(const SpectrumOrTexture& eta)
{
  m_eta = eta;
}

void hikari::BsdfRoughConductor::setK(const SpectrumOrTexture& k)
{
  m_k = k;
}

auto hikari::BsdfRoughConductor::getEta() const -> SpectrumOrTexture
{
  return m_eta;
}

auto hikari::BsdfRoughConductor::getK() const -> SpectrumOrTexture
{
  return m_k;
}

auto hikari::BsdfRoughConductor::getSpecularReflectance() const -> SpectrumOrTexture
{
  return m_specular_reflectance;
}

void hikari::BsdfRoughConductor::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

void hikari::BsdfRoughConductor::setDistribution(BsdfDistributionType distribution)
{
  m_distribution = distribution;
}

auto hikari::BsdfRoughConductor::getDistribution() const -> BsdfDistributionType
{
  return m_distribution;
}

void hikari::BsdfRoughConductor::setAlpha(const FloatOrTexture& alpha)
{
  m_anisotropy = false;
  m_alpha_u = alpha;
  m_alpha_v = alpha;
}

auto hikari::BsdfRoughConductor::getAlpha() const -> std::optional<FloatOrTexture>
{
  if (!m_anisotropy) {
    return m_alpha_u;
  }
  else {
    return std::nullopt;
  }
}


hikari::BsdfRoughConductor::BsdfRoughConductor() :Bsdf(),
m_eta{},
m_k{},
m_specular_reflectance{ SpectrumUniform::create(1.0f) },
m_distribution{BsdfDistributionType::eBeckman},
m_anisotropy{false},
m_alpha_u{ 0.1f },
m_alpha_v{ 0.1f }
{
}
auto hikari::BsdfRoughConductor::getAlphaU() const->FloatOrTexture
{
  return m_alpha_u;
}
void hikari::BsdfRoughConductor::setAlphaU(const FloatOrTexture& alpha_u)
{
  m_anisotropy = true;
  m_alpha_u = alpha_u;
}

auto hikari::BsdfRoughConductor::getAlphaV() const->FloatOrTexture
{
  return m_alpha_v;
}
void hikari::BsdfRoughConductor::setAlphaV(const FloatOrTexture& alpha_v)
{
  m_anisotropy = true;
  m_alpha_v = alpha_v;
}
