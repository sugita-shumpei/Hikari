#pragma once
#include <hikari/core/camera.h>
namespace hikari {
  struct CameraOrthographic : public Camera{
    static constexpr Uuid ID() { return Uuid::from_string("919E233D-AFC5-40FF-97FE-23FFFD3E19CB").value(); }
    static auto create() -> std::shared_ptr<CameraOrthographic>;
    virtual ~CameraOrthographic();
    Uuid getID() const override;
    F32  getNearClip() const;
    F32  getFarClip() const;
    void setNearClip(F32 near_clip);
    void setFarClip(F32 far_clip);
    auto getProjMatrix() const->Mat4x4;
  private:
    CameraOrthographic();
  private:
    F32 m_near_clip;
    F32 m_far_clip;
  };
}
