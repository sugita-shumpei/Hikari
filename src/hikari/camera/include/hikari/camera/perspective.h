#pragma once
#include <hikari/core/camera.h>
#include <optional>
namespace hikari {
  struct CameraPerspective : public Camera {
    static constexpr Uuid ID() { return Uuid::from_string("ECE3495D-B220-4986-8AB0-EC7F7A0D1674").value(); }
    static auto create() -> std::shared_ptr<CameraPerspective>;
    virtual ~CameraPerspective();
    Uuid getID() const override;
    F32  getNearClip() const;
    F32  getFarClip() const;
    void setNearClip(F32 near_clip);
    void setFarClip(F32 far_clip);
    void setFov(F32  fov);
    auto getFov()const -> std::optional<F32>;
    void setFocalLength(F32 focal_length);
    auto getFocalLength() const ->std::optional<F32>;
    void setFovAxis(CameraFovAxis fov_axis);
    auto getFovAxis()const->CameraFovAxis;
  private:
    CameraPerspective();
  private:
    enum class Config {
      eUseFocalLength,
      eUseFov
    };
    F32           m_near_clip;
    F32           m_far_clip;
    F32           m_fov_or_focal_length;
    CameraFovAxis m_fov_axis;
    Config        m_config;
  };
}
