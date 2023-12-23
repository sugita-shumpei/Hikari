#pragma once
#include <hikari/core/light.h>
#include <hikari/core/shape.h>
#include <hikari/core/bitmap.h>
namespace hikari {
  struct Node;
  struct LightEnvmap : public Light {
    static constexpr Uuid ID() { return Uuid::from_string("3AC22ECE-EEF1-4CC9-A876-09B71C096672").value(); }
    static auto create() -> std::shared_ptr<LightEnvmap>;
    virtual ~LightEnvmap();
    Uuid getID() const override;

    void setBitmap(const BitmapPtr& bitmap);
    auto getBitmap()const->BitmapPtr;
    void setScale(F32 scale);
    auto getScale() const->F32;
  private:
    LightEnvmap();
  private:
    BitmapPtr m_bitmap;
    F32       m_scale ;
  };
}
