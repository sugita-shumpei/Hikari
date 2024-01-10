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
  m_int_fresnel_diffuse_reflectance = CalculateFresnelDiffuseReflectance(1.0f/getEta());
  m_ext_fresnel_diffuse_reflectance = CalculateFresnelDiffuseReflectance(getEta());
}

void hikari::BsdfPlastic::setExtIOR(F32 ext_ior)
{
  m_ext_ior = ext_ior;
  m_int_fresnel_diffuse_reflectance = CalculateFresnelDiffuseReflectance(1.0f / getEta());
  m_ext_fresnel_diffuse_reflectance = CalculateFresnelDiffuseReflectance(getEta());
}

hikari::F32 hikari::BsdfPlastic::getIntIOR() const
{
  return m_int_ior;
}
hikari::F32 hikari::BsdfPlastic::getExtIOR() const
{
    return m_ext_ior;
}

hikari::F32 hikari::BsdfPlastic::getEta() const
{
  return m_int_ior/m_ext_ior;
}

hikari::F32 hikari::BsdfPlastic::getIntFresnelDiffuseReflectance() const
{
  return m_int_fresnel_diffuse_reflectance;
}

hikari::F32 hikari::BsdfPlastic::getExtFresnelDiffuseReflectance() const
{
  return m_ext_fresnel_diffuse_reflectance;
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
  m_specular_reflectance{ SpectrumUniform::create(1.0f) },
  m_ext_ior{ 1.0f }, m_int_ior{ 1.0f },m_int_fresnel_diffuse_reflectance{0.0f}, m_ext_fresnel_diffuse_reflectance{ 0.0f },
  m_nonlinear{ false } {
}

auto hikari::BsdfPlastic::CalculateFresnelDiffuseReflectance(F32 eta) -> F32
{
  constexpr size_t N = 10000;
  float d_t = (1.0f/ (float)N);
  float res = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float ti_0 = i * d_t;
    float ti_1 = (i + 1) * d_t;
    float si_0 = std::sqrtf(1.0f- ti_0);
    float ci_0 = std::sqrtf(ti_0);
    float si_1 = std::sinf(1.0f - ti_1);
    float ci_1 = std::sinf(ti_1);
    float fi_0 = calculateReflection(si_0, ci_0, eta);
    float fi_1 = calculateReflection(si_1, ci_1, eta);
    res += (fi_0 + fi_1) * d_t * 0.5f;
  }
  return res;
}

auto hikari::BsdfPlastic::calculateReflection(F32 sin_in, F32 cos_in, F32 eta) -> F32
{
  if (eta == 0.0f) { return 1.0f; }
  float sin_out = sin_in * eta;
  if (sin_out > 1.0f) { return 1.0f; }
  float cos_out = sqrt(fmaxf(1.0f - sin_out * sin_out, 0.0f));
  float rs = (cos_in - cos_out * eta) / (cos_in + cos_out * eta);
  float rp = (eta * cos_in - cos_out) / (eta * cos_in + cos_out);
  return (rp * rp + rs * rs) * 0.5f;
}

hikari::Uuid hikari::BsdfPlastic::getID() const
{
  return ID();
}
