#include <hikari/core/graphics/opengl/context.h>
#include <hikari/core/graphics/opengl/renderer.h>

hikari::core::GraphicsOpenGLContext::~GraphicsOpenGLContext() noexcept
{
  m_handle = nullptr;
}

void hikari::core::GraphicsOpenGLContext::registerThread()
{
  if (m_thid == std::thread::id{}) {
    m_thid = std::this_thread::get_id();
  }
}

auto hikari::core::GraphicsOpenGLContext::getThreadID() const noexcept -> std::thread::id
{
    return m_thid;
}

void hikari::core::GraphicsOpenGLContext::setCurrent()
{
  assert(isOwnerThread(std::this_thread::get_id()));
  m_renderer->setContextCurrent(m_handle);
}

void hikari::core::GraphicsOpenGLContext::popCurrent()
{
  assert(isOwnerThread(std::this_thread::get_id()));
  m_renderer->setContextCurrent(nullptr);
}

auto hikari::core::GraphicsOpenGLContext::getHandle() const-> void*
{
    return m_handle;
}

hikari::core::GraphicsOpenGLContext::GraphicsOpenGLContext(GraphicsOpenGLRenderer* renderer, void* handle)
  :m_renderer{renderer},m_handle{handle}
{
}
