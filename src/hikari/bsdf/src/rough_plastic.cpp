#include <hikari/bsdf/rough_plastic.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfRoughPlastic::create() -> std::shared_ptr<BsdfRoughPlastic>
{
  return std::shared_ptr<BsdfRoughPlastic>(new BsdfRoughPlastic());
}

hikari::BsdfRoughPlastic::~BsdfRoughPlastic()
{
}

void hikari::BsdfRoughPlastic::setIntIOR(F32 int_ior)
{
  m_int_ior = int_ior;
}

void hikari::BsdfRoughPlastic::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
}

hikari::F32 hikari::BsdfRoughPlastic::getIntIOR() const
{
  return m_int_ior;
}
hikari::F32 hikari::BsdfRoughPlastic::getExtIOR() const
{
  return m_ext_ior;
}

auto hikari::BsdfRoughPlastic::getDiffuseReflectance() const -> SpectrumOrTexture
{
  return m_diffuse_reflectance;
}

void hikari::BsdfRoughPlastic::setDiffuseReflectance(const SpectrumOrTexture& ref)
{
  m_diffuse_reflectance = ref;
}

auto hikari::BsdfRoughPlastic::getSpecularReflectance() const -> SpectrumOrTexture
{
  return m_specular_reflectance;
}

void hikari::BsdfRoughPlastic::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

hikari::Bool hikari::BsdfRoughPlastic::getNonLinear() const
{
  return m_nonlinear;
}

void hikari::BsdfRoughPlastic::setNonLinear(Bool non_linear)
{
  m_nonlinear = non_linear;
}

hikari::BsdfRoughPlastic::BsdfRoughPlastic() :Bsdf(),
m_diffuse_reflectance{ SpectrumUniform::create(0.5f) },
m_specular_reflectance{ SpectrumUniform::create(0.5f) },
m_ext_ior{ 1.0f }, m_int_ior{ 1.0f },
m_nonlinear{ false } {
}

hikari::Uuid hikari::BsdfRoughPlastic::getID() const
{
  return ID();
}


void hikari::BsdfRoughPlastic::setDistribution(BsdfDistributionType distribution)
{
  m_distribution = distribution;
}

auto hikari::BsdfRoughPlastic::getDistribution() const -> BsdfDistributionType
{
  return m_distribution;
}

void hikari::BsdfRoughPlastic::setAlpha(const FloatOrTexture& alpha)
{
  m_anisotropy = false;
  m_alpha_u = alpha;
  m_alpha_v = alpha;
}

auto hikari::BsdfRoughPlastic::getAlpha() const -> std::optional<FloatOrTexture>
{
  if (!m_anisotropy) { return m_alpha_u; }
  else { return std::nullopt; }
}

auto hikari::BsdfRoughPlastic::getAlphaU() const->FloatOrTexture
{
  return m_alpha_u;
}
void hikari::BsdfRoughPlastic::setAlphaU(const FloatOrTexture& alpha_u)
{
  m_anisotropy = true;
  m_alpha_u = alpha_u;
}

auto hikari::BsdfRoughPlastic::getAlphaV() const->FloatOrTexture
{
  return m_alpha_v;
}
void hikari::BsdfRoughPlastic::setAlphaV(const FloatOrTexture& alpha_v)
{
  m_anisotropy = true;
  m_alpha_v = alpha_v;
}
