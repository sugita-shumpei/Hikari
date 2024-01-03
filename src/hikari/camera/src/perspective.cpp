#include <hikari/camera/perspective.h>
#include <hikari/core/film.h>
#include <glm/gtx/transform.hpp>

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

auto hikari::CameraPerspective::getProjMatrix() const -> Mat4x4
{
  auto op_fov = this->getFov();
  auto fovy = static_cast<float>(0.0f);

  auto film = getFilm();
  if (!film) {
    throw std::runtime_error("Film Must Be Attached!");
  }

  auto w = film->getWidth();
  auto h = film->getHeight();

  auto aspect = static_cast<float>(w) / static_cast<float>(h);
  if (op_fov) {
    auto axis = this->getFovAxis();
    if (axis == hikari::CameraFovAxis::eSmaller) {
      if (aspect > 1.0f) {// W/H > 1.0f
        axis = hikari::CameraFovAxis::eY;
      }
      else {
        axis = hikari::CameraFovAxis::eX;
      }
    }
    if (axis == hikari::CameraFovAxis::eLarger) {
      if (aspect > 1.0f) {// W/H > 1.0f
        axis = hikari::CameraFovAxis::eX;
      }
      else {
        axis = hikari::CameraFovAxis::eY;
      }
    }

    if (axis == hikari::CameraFovAxis::eX) {
      ///Y |         |X
      ///H |         |
      ///  |_______  |_______Z
      ///     W    X
      auto ax = tanf(0.5f * glm::radians(*op_fov));
      auto ay = ax / aspect;
      fovy = 2.0f * atanf(ay);
    }
    else if (axis == hikari::CameraFovAxis::eY) {
      fovy = glm::radians(*op_fov);
    }
    else {
      throw std::runtime_error("Unsupported Axis Type!");
    }
  }
  else {
    throw std::runtime_error("Unsupported Camera!");
  }

  return glm::perspective(fovy, aspect, m_near_clip, m_far_clip);
}
auto hikari::CameraPerspective::getProjMatrix_Infinite() const -> Mat4x4
{
  auto op_fov = this->getFov();
  auto fovy = static_cast<float>(0.0f);

  auto film = getFilm();
  if (!film) {
    throw std::runtime_error("Film Must Be Attached!");
  }

  auto w = film->getWidth();
  auto h = film->getHeight();

  auto aspect = static_cast<float>(w) / static_cast<float>(h);
  if (op_fov) {
    auto axis = this->getFovAxis();
    if (axis == hikari::CameraFovAxis::eSmaller) {
      if (aspect > 1.0f) {// W/H > 1.0f
        axis = hikari::CameraFovAxis::eY;
      }
      else {
        axis = hikari::CameraFovAxis::eX;
      }
    }
    if (axis == hikari::CameraFovAxis::eLarger) {
      if (aspect > 1.0f) {// W/H > 1.0f
        axis = hikari::CameraFovAxis::eX;
      }
      else {
        axis = hikari::CameraFovAxis::eY;
      }
    }

    if (axis == hikari::CameraFovAxis::eX) {
      ///Y |         |X
      ///H |         |
      ///  |_______  |_______Z
      ///     W    X
      auto ax = tanf(0.5f * glm::radians(*op_fov));
      auto ay = ax / aspect;
      fovy = 2.0f * atanf(ay);
    }
    else if (axis == hikari::CameraFovAxis::eY) {
      fovy = glm::radians(*op_fov);
    }
    else {
      throw std::runtime_error("Unsupported Axis Type!");
    }
  }
  else {
    throw std::runtime_error("Unsupported Camera!");
  }

  return glm::infinitePerspective(fovy, aspect, m_near_clip);
}

hikari::CameraPerspective::CameraPerspective()
  :m_near_clip{ 1e-2f }, m_far_clip{ 1e4f },m_config{Config::eUseFocalLength},m_fov_or_focal_length{50.0f},m_fov_axis{CameraFovAxis::eX}
{
}
