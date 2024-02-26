#include <hikari/core/device/window.h>
using namespace hikari;

void hikari::Window::setSize(UVec2 size)
{
//  m_size = size;
}

UVec2 hikari::Window::getSize() const
{
  return m_size;
}

void hikari::Window::setPosition(IVec2 pos)
{
//  m_position = pos;
}

IVec2 hikari::Window::getPosition() const
{
  return m_position;
}

UVec2 hikari::Window::getFrameBufferSize() const
{
  return m_frame_buffer_size;
}

DVec2 hikari::Window::getCursorPosition() const
{
  return m_cursor_position;
}
