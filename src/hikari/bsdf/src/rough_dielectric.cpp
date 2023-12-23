#include <hikari/bsdf/rough_dielectric.h>
#include <hikari/spectrum/uniform.h>

auto hikari::BsdfRoughDielectric::getSpecularReflectance() const -> SpectrumOrTexture
{
  return m_specular_reflectance;
}

void hikari::BsdfRoughDielectric::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

auto hikari::BsdfRoughDielectric::getSpecularTransmittance() const -> SpectrumOrTexture
{
  return m_specular_transmittance;
}

void hikari::BsdfRoughDielectric::setSpecularTransmittance(const SpectrumOrTexture& tra)
{
  m_specular_transmittance = tra;
}

void hikari::BsdfRoughDielectric::setDistribution(BsdfDistributionType distribution)
{
  m_distribution = distribution;
}

auto hikari::BsdfRoughDielectric::getDistribution() const -> BsdfDistributionType
{
  return m_distribution;
}

void hikari::BsdfRoughDielectric::setAlpha(const FloatOrTexture& alpha)
{
  m_anisotropy = false;
  m_alpha_u    = alpha;
  m_alpha_v    = alpha;
}

auto hikari::BsdfRoughDielectric::getAlpha() const -> std::optional<FloatOrTexture>
{
  if (!m_anisotropy) { return m_alpha_u; }
  else { return std::nullopt; }
}

hikari::BsdfRoughDielectric::BsdfRoughDielectric(F32 int_ior, F32 ext_ior) :Bsdf(),
m_int_ior{ int_ior },
m_ext_ior{ ext_ior },
m_specular_reflectance  { SpectrumUniform::create(1.0f) },
m_specular_transmittance{ SpectrumUniform::create(1.0f) },
m_distribution{BsdfDistributionType::eBeckman},
m_alpha_u{0.1f}, m_alpha_v{ 0.1f }, m_anisotropy{false}
{
}

auto hikari::BsdfRoughDielectric::create(F32 int_ior, F32 ext_ior) -> std::shared_ptr<BsdfRoughDielectric>
{
  return std::shared_ptr<BsdfRoughDielectric>(new BsdfRoughDielectric(int_ior, ext_ior));
}

hikari::BsdfRoughDielectric::~BsdfRoughDielectric()
{
}

hikari::Uuid hikari::BsdfRoughDielectric::getID() const { return ID(); }

void hikari::BsdfRoughDielectric::setIntIOR(F32 int_ior)
{
  m_int_ior = int_ior;
}

void hikari::BsdfRoughDielectric::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
}

hikari::F32 hikari::BsdfRoughDielectric::getIntIOR() const
{
  return m_int_ior;
}

hikari::F32 hikari::BsdfRoughDielectric::getExtIOR() const
{
  return m_ext_ior;
}

auto hikari::BsdfRoughDielectric::getAlphaU() const->FloatOrTexture
{
  return m_alpha_u;
}
void hikari::BsdfRoughDielectric::setAlphaU(const FloatOrTexture& alpha_u)
{
  m_anisotropy = true;
  m_alpha_u = alpha_u;
}

auto hikari::BsdfRoughDielectric::getAlphaV() const->FloatOrTexture
{
  return m_alpha_v;
}
void hikari::BsdfRoughDielectric::setAlphaV(const FloatOrTexture& alpha_v)
{
  m_anisotropy = true;
  m_alpha_v = alpha_v;
}
