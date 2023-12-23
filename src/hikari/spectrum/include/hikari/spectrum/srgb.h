#pragma once
#include <hikari/core/spectrum.h>
#include <hikari/core/color.h>
namespace hikari {
    // レンダラー指定のSRGBカラースペースとして, 値を設定する.
    // XYZへの変換には色空間を指定すること
  struct SpectrumSrgb : public Spectrum {
    static constexpr Uuid ID() { return Uuid::from_string("0D1F4DA7-25A9-487C-A416-DBA7D14C8EB2").value(); }
    static auto create(const Rgb3F& color = {1.0f,1.0f,1.0f}) -> std::shared_ptr<SpectrumSrgb>;
    virtual ~SpectrumSrgb();
    Uuid getID() const override;
    void setColor(const Rgb3F& color);
    auto getColor() const->Rgb3F;
  private:
    SpectrumSrgb(const Rgb3F& color);
  private:
    Rgb3F m_color;
  };
}
