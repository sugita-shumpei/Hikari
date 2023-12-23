#include <hikari/camera/orthographic.h>

auto hikari::CameraOrthographic::create() -> std::shared_ptr<CameraOrthographic>
{
  return std::shared_ptr<CameraOrthographic>(new CameraOrthographic());
}

hikari::CameraOrthographic::~CameraOrthographic()
{
}

hikari::Uuid hikari::CameraOrthographic::getID() const
{
    return ID();
}

hikari::F32 hikari::CameraOrthographic::getNearClip() const
{
  return m_near_clip;
}

hikari::F32 hikari::CameraOrthographic::getFarClip() const
{
  return m_far_clip;
}

void hikari::CameraOrthographic::setNearClip(F32 near_clip)
{
  m_near_clip = near_clip;
}

void hikari::CameraOrthographic::setFarClip(F32 far_clip)
{
  m_far_clip;
}

hikari::CameraOrthographic::CameraOrthographic()
  :m_near_clip{1e-2f},m_far_clip{1e4f}
{
}
