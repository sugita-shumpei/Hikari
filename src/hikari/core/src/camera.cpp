#include <hikari/core/camera.h>
#include <hikari/core/node.h>
#include <hikari/core/medium.h>
#include <hikari/core/film.h>

hikari::Camera::~Camera() noexcept
{
}

auto hikari::Camera::getNode() -> std::shared_ptr<Node> { return m_node.lock(); }

auto hikari::Camera::getFilm() -> std::shared_ptr<Film>
{
    return m_film;
}

void hikari::Camera::setFilm(const std::shared_ptr<Film>& film)
{
  m_film = film;
}

void hikari::Camera::setMedium(const std::shared_ptr<Medium>& medium)
{
  m_medium = medium;
}

auto hikari::Camera::getMedium() -> std::shared_ptr<Medium>
{
    return m_medium;
}

hikari::Camera::Camera() :m_node{}, m_medium{}
{
}

void hikari::Camera::onAttach(const std::shared_ptr<Node>& node)
{
  m_node = node;
}

void hikari::Camera::onDetach()
{
  m_node = {};
}
