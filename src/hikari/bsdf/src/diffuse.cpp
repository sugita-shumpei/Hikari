#include <hikari/bsdf/diffuse.h>
#include <hikari/spectrum/uniform.h>
auto hikari::BsdfDiffuse::create() -> std::shared_ptr<BsdfDiffuse> {
  return std::shared_ptr<BsdfDiffuse>(new hikari::BsdfDiffuse(SpectrumUniform::create(0.5f)));
}
auto hikari::BsdfDiffuse::create(const SpectrumOrTexture& ref = {}) -> std::shared_ptr<BsdfDiffuse> {
  return std::shared_ptr<BsdfDiffuse>(new hikari::BsdfDiffuse(ref));
}
hikari::BsdfDiffuse::~BsdfDiffuse() {}
hikari::Uuid hikari::BsdfDiffuse::getID() const { return ID(); }
auto hikari::BsdfDiffuse::getReflectance() const->SpectrumOrTexture { return m_reflectance; }
void hikari::BsdfDiffuse::setReflectance(const SpectrumOrTexture& ref) {
  m_reflectance = ref;
}
hikari::BsdfDiffuse::BsdfDiffuse(const SpectrumOrTexture& ref) :Bsdf(), m_reflectance{ref} {}
