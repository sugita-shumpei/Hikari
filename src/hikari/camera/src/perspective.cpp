#include <hikari/camera/perspective.h>

auto hikari::CameraPerspective::create() -> std::shared_ptr<CameraPerspective>
{
  return std::shared_ptr<CameraPerspective>(new CameraPerspective());
}

hikari::CameraPerspective::~CameraPerspective()
{
}

hikari::Uuid hikari::CameraPerspective::getID() const
{
  return ID();
}

hikari::F32 hikari::CameraPerspective::getNearClip() const
{
  return m_near_clip;
}

hikari::F32 hikari::CameraPerspective::getFarClip() const
{
  return m_far_clip;
}

void hikari::CameraPerspective::setNearClip(F32 near_clip)
{
  m_near_clip = near_clip;
}

void hikari::CameraPerspective::setFarClip(F32 far_clip)
{
  m_far_clip;
}

void hikari::CameraPerspective::setFov(F32 fov)
{
  m_config = Config::eUseFov;
  m_fov_or_focal_length = fov;
}

auto hikari::CameraPerspective::getFov() const -> std::optional<F32>
{
  if (m_config == Config::eUseFov) { return m_fov_or_focal_length; }
  else { return std::nullopt; }
}

void hikari::CameraPerspective::setFocalLength(F32 focal_length)
{
  m_config = Config::eUseFocalLength;
  m_fov_or_focal_length = focal_length;
  
}

auto hikari::CameraPerspective::getFocalLength() const -> std::optional<F32>
{
  if (m_config == Config::eUseFocalLength) { return m_fov_or_focal_length; }
  else { return std::nullopt; }
}

void hikari::CameraPerspective::setFovAxis(CameraFovAxis fov_axis)
{
  m_fov_axis = fov_axis;
}

auto hikari::CameraPerspective::getFovAxis() const -> CameraFovAxis
{
  return m_fov_axis;
}

hikari::CameraPerspective::CameraPerspective()
  :m_near_clip{ 1e-2f }, m_far_clip{ 1e4f },m_config{Config::eUseFocalLength},m_fov_or_focal_length{50.0f},m_fov_axis{CameraFovAxis::eX}
{
}
