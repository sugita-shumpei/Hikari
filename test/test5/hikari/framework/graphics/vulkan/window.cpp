#include <hikari/platform/vulkan/render/gfx_vulkan_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
#include <hikari/platform/vulkan/render/gfx_vulkan_instance.h>
auto hikari::platforms::vulkan::render::GFXVulkanWindowObject::create(const std::shared_ptr<GFXVulkanInstanceObject>& instance, const GFXWindowDesc& desc) -> std::shared_ptr<GFXVulkanWindowObject> {

  auto& manager = glfw::WindowManager::getInstance();
  int x = desc.x == UINT32_MAX ? -1 : desc.x;
  int y = desc.y == UINT32_MAX ? -1 : desc.y;
  auto wnd = manager.createVulkanWindow(desc.width, desc.height, x, y, desc.title, desc.is_visible, desc.is_resizable,desc.is_borderless);
  if (!wnd) { return nullptr; }
  auto res = std::shared_ptr<GFXVulkanWindowObject>(new GFXVulkanWindowObject(instance,(GLFWwindow*)wnd, desc));
  if (res) { res->initEvents(); }
  return res;
}

hikari::platforms::vulkan::render::GFXVulkanWindowObject::~GFXVulkanWindowObject() noexcept {
  if (m_instance) {
    auto vk_instance = m_instance->getVKInstance();
    if (m_vk_surface && m_vkDestroySurfaceKHR) {
      m_vkDestroySurfaceKHR(vk_instance, m_vk_surface, nullptr);
    }

  }
}

auto hikari::platforms::vulkan::render::GFXVulkanWindowObject::getVKSurface() const -> VkSurfaceKHR { return m_vk_surface; }


hikari::platforms::vulkan::render::GFXVulkanWindowObject::GFXVulkanWindowObject(const std::shared_ptr<GFXVulkanInstanceObject>& instance, GLFWwindow* window, const GFXWindowDesc& desc) :
  hikari::platforms::glfw::render::GFXGLFWWindowObject(window,desc), m_instance{ instance },m_vk_surface{ nullptr }, m_vkDestroySurfaceKHR{ nullptr }, m_visible{desc.is_visible} {
  if (instance) {
    auto& manager = glfw::WindowManager::getInstance();
    int x = desc.x == UINT32_MAX ? -1 : desc.x;
    int y = desc.y == UINT32_MAX ? -1 : desc.y;
    auto window = getHandle();
    m_vk_surface = reinterpret_cast<VkSurfaceKHR>(manager.createVulkanSurface(window, m_instance->getVKInstance()));
    m_vkDestroySurfaceKHR = instance->getVKInstanceProcAddr<PFN_vkDestroySurfaceKHR>("vkDestroySurfaceKHR");
  }
}

