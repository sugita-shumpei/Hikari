#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/color.h>
namespace hikari {
  // [min,max]間で一様なスペクトルを定義
  // RGBレンダリングの場合,
  // 反射率の設定時には, SRGB(m_value,m_value,m_value)へ変換
  // 発光度の設定時には,  XYZ(m_value,m_value,m_value)へ変換
  struct SpectrumUniform : public Spectrum {
    static constexpr Uuid ID() { return Uuid::from_string("0C75FBB3-D5DA-48C1-B3B2-8E12233D6E22").value(); }
    static auto create(F32 value = 1.0f) -> std::shared_ptr<SpectrumUniform>;
    virtual ~SpectrumUniform();
    Uuid getID() const override;

    void setValue(F32 value);
    void setMinWavelength(F32 wavelength);
    void setMaxWavelength(F32 wavelength);
    auto getValue()const->F32;
    auto getMinWavelength() const->F32;
    auto getMaxWavelength() const->F32;
  private:
    SpectrumUniform(F32 value);
  private:
    F32 m_value;
    F32 m_min_wavelength;
    F32 m_max_wavelength;
  };

}
