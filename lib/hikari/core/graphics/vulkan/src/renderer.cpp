#include <hikari/core/graphics/vulkan/renderer.h>

hikari::core::GraphicsVulkanRenderer::GraphicsVulkanRenderer(Window* window)
  :hikari::core::WindowRenderer(window)
{
}


hikari::core::GraphicsVulkanRenderer::~GraphicsVulkanRenderer() noexcept {}

bool hikari::core::GraphicsVulkanRenderer::initialize()
{
  if (m_is_init) { return true; }
  if (!initInstance()) { return false; }
  m_is_init_instance = true;
  if (!initSurface()) { return false; }
  m_is_init_surface = true;
  if (!initDevice()) { return false; }
  m_is_init_device = true;
  if (!initCustom()) { return false; }
  m_is_init_extra = true;
  m_is_init = true;
  return true;
}

void hikari::core::GraphicsVulkanRenderer::terminate()
{
  freeCustom();
  freeDevice();
  freeSurface();
  freeInstance();
  m_is_init = false;

}

auto hikari::core::GraphicsVulkanRenderer::getInstance() -> GraphicsVulkanInstance&
{
  return *m_instance;
}

auto hikari::core::GraphicsVulkanRenderer::getDevice() -> GraphicsVulkanDevice&
{
  // TODO: return ステートメントをここに挿入します
  return *m_device;
}

void hikari::core::GraphicsVulkanRenderer::setInstance(std::unique_ptr<GraphicsVulkanInstance>&& instance)
{
  if (m_is_init_instance) { return; }
  m_instance = std::move(instance);
}

void hikari::core::GraphicsVulkanRenderer::setDevice(std::unique_ptr<GraphicsVulkanDevice>&& device)
{
  if (m_is_init_device) { return; }
  m_device = std::move(device);
}

bool hikari::core::GraphicsVulkanRenderer::initInstance()
{
  auto instance = std::make_unique<GraphicsVulkanInstance>();
  instance->requestExtension("VK_KHR_surface");
  instance->requestExtension("VK_KHR_win32_surface");
  instance->requestExtension("VK_EXT_debug_utils");
  instance->requestLayer("VK_LAYER_KHRONOS_validation");
  if (instance->create()) {
    setInstance(std::move(instance));
    return true;
  }
  return false;
}

bool hikari::core::GraphicsVulkanRenderer::initSurface()
{
  auto surface = (VkSurfaceKHR)vkCreateSurface(*getInstance());
  if (surface) {
    m_surface.reset(new vk::raii::SurfaceKHR(static_cast<vk::raii::Instance&>(getInstance()),surface));
    return true;
  }
  return false;
}

bool hikari::core::GraphicsVulkanRenderer::initDevice()
{
  auto device = std::make_unique<GraphicsVulkanDevice>(getInstance());
  device->requestExtension("VK_KHR_swapchain");
  if (device->create()) {
    setDevice(std::move(device));
    return true;
  }
  return false;
}

bool hikari::core::GraphicsVulkanRenderer::initCustom()
{
  return true;
}

void hikari::core::GraphicsVulkanRenderer::freeInstance()
{
  if (m_is_init_instance){
    m_instance.reset();
  }
  m_is_init_instance = false;
}

void hikari::core::GraphicsVulkanRenderer::freeSurface()
{
  if (m_is_init_surface) {
    m_surface.reset();
  }
  m_is_init_surface = false;

}

void hikari::core::GraphicsVulkanRenderer::freeDevice()
{
  if (m_is_init_device) {
    m_device.reset();
  }
  m_is_init_device = false;
}

void hikari::core::GraphicsVulkanRenderer::freeCustom()
{
  m_is_init_extra = false;
}
