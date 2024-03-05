#include <hikari/core/graphics/opengl/renderer.h>
#include <hikari/core/graphics/opengl/context.h>
#include <hikari/core/window/window.h>

hikari::core::GraphicsOpenGLRenderer::GraphicsOpenGLRenderer(Window* window)
  :WindowRenderer(window), m_render_context{nullptr}
{
}

bool hikari::core::GraphicsOpenGLRenderer::initialize()
{
  if (m_is_init) { return true; }
  if (!initContext()) { return false; }
  if (!initCustom()) { return false; }
  m_is_init_extra = true;
  m_is_init = true;
  return true;
}

void hikari::core::GraphicsOpenGLRenderer::terminate()
{
  freeCustom();
  m_is_init_extra = false;
  freeContext();
  m_is_init = false;
}

void hikari::core::GraphicsOpenGLRenderer::update()
{
  auto main_context = getRenderContext();
  main_context->setCurrent();
  updateCustom();
  swapBuffers();
}

auto hikari::core::GraphicsOpenGLRenderer::getRenderContext() -> GraphicsOpenGLContext*
{
  return m_render_context.get();
}

auto hikari::core::GraphicsOpenGLRenderer::createSharingContext() -> GraphicsOpenGLContext*
{
    return new GraphicsOpenGLContext(this,glCreateSharingContext());
}

void hikari::core::GraphicsOpenGLRenderer::destroySharingContext(GraphicsOpenGLContext* context)
{
  if (context) {
    glDestroySharingContext(context->getHandle());
    delete context;
  }
}

bool hikari::core::GraphicsOpenGLRenderer::initContext()
{
  if (m_is_init_context) { return m_is_init_context; }
  m_render_context.reset(new GraphicsOpenGLContext(this, getWindow()->getNativeHandle()));
  std::future<void> future = executeInRenderThread<std::function<void(void)>, void>([this]()->void {
    auto main_context = getRenderContext();
    main_context->registerThread();
    main_context->setCurrent();
  });
  future.wait();

  m_window_context.reset(createSharingContext());
  m_window_context->registerThread();
  m_window_context->setCurrent();
  if (!gladLoadGLLoader((GLADloadproc)glGetProcAddress())) {
    m_window_context->popCurrent();
    m_window_context.reset();
    m_render_context.reset();
    return false;
  }
  m_is_init_context = true;
  return true;
}

void hikari::core::GraphicsOpenGLRenderer::freeContext()
{
  auto future = executeInRenderThread<std::function<void(void)>,void>([this]()->void {
    auto main_context = getRenderContext();
    main_context->popCurrent();
    return;
  });
  future.wait();
  m_window_context.reset();
  m_render_context.reset();
}

bool hikari::core::GraphicsOpenGLRenderer::initCustom()
{
    return true;
}

void hikari::core::GraphicsOpenGLRenderer::freeCustom()
{
}

void hikari::core::GraphicsOpenGLRenderer::updateCustom()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  auto size = getSurfaceSize();
  glViewport(0, 0, size.width, size.height);
}
