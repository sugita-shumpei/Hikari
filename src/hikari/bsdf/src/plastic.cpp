#include <hikari/bsdf/plastic.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfPlastic::create() -> std::shared_ptr<BsdfPlastic>
{
    return std::shared_ptr<BsdfPlastic>(new BsdfPlastic());
}

hikari::BsdfPlastic::~BsdfPlastic()
{
}

void hikari::BsdfPlastic::setIntIOR(F32 int_ior)
{
  m_int_ior = int_ior;
}

void hikari::BsdfPlastic::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
}

hikari::F32 hikari::BsdfPlastic::getIntIOR() const
{
  return m_int_ior;
}
hikari::F32 hikari::BsdfPlastic::getExtIOR() const
{
    return m_ext_ior;
}

auto hikari::BsdfPlastic::getDiffuseReflectance() const -> SpectrumOrTexture
{
    return m_diffuse_reflectance;
}

void hikari::BsdfPlastic::setDiffuseReflectance(const SpectrumOrTexture& ref)
{
  m_diffuse_reflectance = ref;
}

auto hikari::BsdfPlastic::getSpecularReflectance() const -> SpectrumOrTexture
{
    return m_specular_reflectance;
}

void hikari::BsdfPlastic::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

hikari::Bool hikari::BsdfPlastic::getNonLinear() const
{
    return m_nonlinear;
}

void hikari::BsdfPlastic::setNonLinear(Bool non_linear)
{
  m_nonlinear = non_linear;
}

hikari::BsdfPlastic::BsdfPlastic():Bsdf(),
  m_diffuse_reflectance{ SpectrumUniform::create(0.5f)},
  m_specular_reflectance{ SpectrumUniform::create(0.5f) },
  m_ext_ior{ 1.0f }, m_int_ior{ 1.0f },
  m_nonlinear{ false } {
}

hikari::Uuid hikari::BsdfPlastic::getID() const
{
  return ID();
}
