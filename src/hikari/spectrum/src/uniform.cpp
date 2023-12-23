#include <hikari/spectrum/uniform.h>
auto hikari::SpectrumUniform::create(F32 value) -> std::shared_ptr<SpectrumUniform> { return std::shared_ptr<SpectrumUniform>(new SpectrumUniform(value)); }
hikari::SpectrumUniform::~SpectrumUniform() {}
hikari::Uuid hikari::SpectrumUniform::getID() const { return ID(); }

void hikari::SpectrumUniform::setValue(F32 value) { m_value = value; }
void hikari::SpectrumUniform::setMinWavelength(F32 wavelength) { m_min_wavelength = wavelength; }
void hikari::SpectrumUniform::setMaxWavelength(F32 wavelength) { m_max_wavelength = wavelength; }
auto hikari::SpectrumUniform::getValue()const->F32 { return m_value; }
auto hikari::SpectrumUniform::getMinWavelength() const->F32 { return m_min_wavelength; }
auto hikari::SpectrumUniform::getMaxWavelength() const->F32 { return m_max_wavelength; }
hikari::SpectrumUniform::SpectrumUniform(F32 value) : Spectrum(), m_value{ value }, m_min_wavelength{360.0f}, m_max_wavelength{830.0f} {}
