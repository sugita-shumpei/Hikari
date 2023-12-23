#include <hikari/bsdf/phong.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfPhong::create() -> std::shared_ptr<BsdfPhong>
{
    return std::shared_ptr<BsdfPhong>(new BsdfPhong());
}

hikari::BsdfPhong::~BsdfPhong()
{
}

hikari::Uuid hikari::BsdfPhong::getID() const
{
    return ID();
}

auto hikari::BsdfPhong::getDiffuseReflectance() const -> SpectrumOrTexture
{
    return m_specular_reflectance;
}

void hikari::BsdfPhong::setDiffuseReflectance(const SpectrumOrTexture& ref)
{
  m_diffuse_reflectance = ref;
}

auto hikari::BsdfPhong::getSpecularReflectance() const -> SpectrumOrTexture
{
    return m_specular_reflectance;
}

void hikari::BsdfPhong::setSpecularReflectance(const SpectrumOrTexture& ref)
{
  m_specular_reflectance = ref;
}

auto hikari::BsdfPhong::getExponent() const -> FloatOrTexture
{
    return m_exponent;
}

void hikari::BsdfPhong::setExponent(const FloatOrTexture& exp)
{
  m_exponent = exp;
}

hikari::BsdfPhong::BsdfPhong():Bsdf(),
  m_diffuse_reflectance{SpectrumUniform::create(0.2f)},
  m_specular_reflectance{ SpectrumUniform::create(0.5f) },
  m_exponent{ 30.0f}
{
}
