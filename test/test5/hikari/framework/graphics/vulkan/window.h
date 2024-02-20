#pragma once
#include <hikari/render/gfx_common.h>
#include <hikari/render/gfx_window.h>
#include <hikari/platform/glfw/render/gfx_glfw_window.h>
#include <hikari/platform/vulkan/render/gfx_vulkan_common.h>
namespace hikari {
  inline namespace platforms {
    namespace vulkan {
      inline namespace render {
        struct GFXVulkanInstanceObject;
        struct GFXVulkanWindowObject : public  hikari::platforms::glfw::render::GFXGLFWWindowObject {
          static auto create(const std::shared_ptr<GFXVulkanInstanceObject>& instance, const GFXWindowDesc& desc) -> std::shared_ptr<GFXVulkanWindowObject>;
          virtual ~GFXVulkanWindowObject() noexcept;
          auto getAPI() const->GFXAPI override { return GFXAPI::eVulkan; }
          auto getVKSurface() const->VkSurfaceKHR;
        private:
          GFXVulkanWindowObject(const std::shared_ptr<GFXVulkanInstanceObject>& instance, GLFWwindow* window, const GFXWindowDesc& desc);
        private:
          std::shared_ptr<GFXVulkanInstanceObject> m_instance;
          VkSurfaceKHR m_vk_surface;
          PFN_vkDestroySurfaceKHR m_vkDestroySurfaceKHR;
          bool m_visible;
        };
        struct GFXVulkanWindow : protected GFXWindowImpl<GFXVulkanWindowObject> {
          HK_RENDER_GFX_WINDOW_METHOD_OVERRIDES(GFXWindowImpl<GFXVulkanWindowObject>);
          using type = typename impl_type::type;
          GFXVulkanWindow() noexcept :impl_type() {}
          GFXVulkanWindow(const GFXVulkanWindow& v) noexcept : impl_type(v.getObject()) {}
          GFXVulkanWindow(nullptr_t) noexcept :impl_type(nullptr) {}
          GFXVulkanWindow(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
          template<typename GFXVulkanWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXVulkanWindowLike::type>, nullptr_t > = nullptr>
          GFXVulkanWindow(const GFXVulkanWindowLike& v) noexcept :impl_type(v.getObject()) {}
          GFXVulkanWindow& operator=(const GFXVulkanWindow& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
          GFXVulkanWindow& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
          GFXVulkanWindow& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
          template<typename GFXVulkanWindowLike, std::enable_if_t<std::is_base_of_v<type, typename GFXVulkanWindowLike::type>, nullptr_t > = nullptr>
          GFXVulkanWindow& operator=(const GFXVulkanWindowLike& v) noexcept { setObject(v.getObject()); return*this; }
        };
      }
    }
  }
}
