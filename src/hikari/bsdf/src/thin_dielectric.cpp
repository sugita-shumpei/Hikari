#include <hikari/bsdf/thin_dielectric.h>
#include <hikari/spectrum/uniform.h>


auto hikari::BsdfThinDielectric::getSpecularReflectance() const -> SpectrumOrTexture
{
  return m_specular_reflectance;
}

void hikari::BsdfThinDielectric::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

auto hikari::BsdfThinDielectric::getSpecularTransmittance() const -> SpectrumOrTexture
{
  return m_specular_transmittance;
}

void hikari::BsdfThinDielectric::setSpecularTransmittance(const SpectrumOrTexture& tra)
{
  m_specular_transmittance = tra;
}

hikari::BsdfThinDielectric::BsdfThinDielectric(F32 int_ior, F32 ext_ior) :Bsdf(),
m_int_ior{ int_ior },
m_ext_ior{ ext_ior },
m_specular_reflectance{ SpectrumUniform::create(1.0f) },
m_specular_transmittance{ SpectrumUniform::create(1.0f) }
{
}

auto hikari::BsdfThinDielectric::create(F32 int_ior, F32 ext_ior) -> std::shared_ptr<BsdfThinDielectric>
{
  return std::shared_ptr<BsdfThinDielectric>(new BsdfThinDielectric(int_ior, ext_ior));
}

hikari::BsdfThinDielectric::~BsdfThinDielectric()
{
}

void hikari::BsdfThinDielectric::setIntIOR(F32 int_ior)
{
  m_int_ior = int_ior;
}

void hikari::BsdfThinDielectric::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
}

hikari::F32 hikari::BsdfThinDielectric::getIntIOR() const
{
  return m_int_ior;
}

hikari::F32 hikari::BsdfThinDielectric::getExtIOR() const
{
  return m_ext_ior;
}
