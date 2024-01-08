#include <hikari/bsdf/dielectric.h>
#include <hikari/spectrum/uniform.h>


auto hikari::BsdfDielectric::getSpecularReflectance() const -> SpectrumOrTexture
{
    return m_specular_reflectance;
}

void hikari::BsdfDielectric::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

auto hikari::BsdfDielectric::getSpecularTransmittance() const -> SpectrumOrTexture
{
    return m_specular_transmittance;
}

void hikari::BsdfDielectric::setSpecularTransmittance(const SpectrumOrTexture& tra)
{
  m_specular_transmittance = tra;
}

hikari::BsdfDielectric::BsdfDielectric(F32 int_ior, F32 ext_ior):Bsdf(),
m_int_ior{int_ior},
m_ext_ior{ext_ior},
m_specular_reflectance{SpectrumUniform::create(1.0f)},
m_specular_transmittance{ SpectrumUniform::create(1.0f) }
{
}

auto hikari::BsdfDielectric::create(F32 int_ior, F32 ext_ior) -> std::shared_ptr<BsdfDielectric>
{
  return std::shared_ptr<BsdfDielectric>(new BsdfDielectric(int_ior,ext_ior));
}

hikari::BsdfDielectric::~BsdfDielectric()
{
}

void hikari::BsdfDielectric::setIntIOR(F32 int_ior)
{
  m_int_ior = int_ior;
}

void hikari::BsdfDielectric::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
}

hikari::F32 hikari::BsdfDielectric::getIntIOR() const
{
    return m_int_ior;
}

hikari::F32 hikari::BsdfDielectric::getExtIOR() const
{
    return m_ext_ior;
}

hikari::F32 hikari::BsdfDielectric::getEta() const
{
    return m_int_ior/m_ext_ior;
}
