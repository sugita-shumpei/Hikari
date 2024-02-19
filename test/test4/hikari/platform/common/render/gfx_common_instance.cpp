#include <hikari/platform/common/render/gfx_common_instance.h>
#include <hikari/platform/opengl/render/gfx_opengl_instance.h>
#include <hikari/platform/vulkan/render/gfx_vulkan_instance.h>

auto hikari::platforms::common::render::createGFXInstance(GFXAPI api) -> hikari::render::GFXInstance
{
  if (api == GFXAPI::eOpenGL) { return hikari::platforms::opengl::GFXOpenGLInstance(); }
  if (api == GFXAPI::eVulkan) { return hikari::platforms::vulkan::GFXVulkanInstance(); }
  return hikari::render::GFXInstance();
}
