#pragma once
#include <hikari/render/gfx_common.h>
#include <hikari/render/gfx_instance.h>
#include <hikari/platform/vulkan/render/gfx_vulkan_common.h>
namespace hikari {
  inline namespace platforms {
    namespace vulkan {
      inline namespace render {
        struct GFXVulkanWindowObject;
        struct GFXVulkanInstanceObject : public hikari::render::GFXInstanceObject, public std::enable_shared_from_this<GFXVulkanInstanceObject> {
          static auto create() -> std::shared_ptr< GFXVulkanInstanceObject>;
          virtual ~GFXVulkanInstanceObject() noexcept;
          auto getAPI() const->GFXAPI override { return GFXAPI::eVulkan; }

          auto createVulkanWindow(const GFXWindowDesc& desc) -> std::shared_ptr<GFXVulkanWindowObject>;
          auto createWindow(const GFXWindowDesc& desc) -> std::shared_ptr<hikari::render::GFXWindowObject> override;
          auto createDevice(const GFXDeviceDesc& desc, const std::shared_ptr<GFXWindowObject>& window = nullptr) -> std::shared_ptr<GFXDeviceObject> override { return nullptr; }

          auto getVKInstance()            const -> VkInstance;
          auto getVKDebugUtilsMessenger() const -> VkDebugUtilsMessengerEXT;
          template<typename VulkanFunctionT>
          auto getVKInstanceProcAddr(const char* name) const ->VulkanFunctionT;

        private:
          GFXVulkanInstanceObject() noexcept;
          bool initVulkanInstance();
          void freeVulkanInstance();
        private:
          std::unique_ptr<vk::DynamicLoader>  m_dll;
          VkInstance                          m_vk_instance;
#ifndef NDEBUG
          VkDebugUtilsMessengerEXT            m_vk_debug_utils_messenger;
#endif
          uint32_t                            m_vk_instance_version;
          std::vector<vk::ExtensionProperties>m_vk_instance_exten_properties;
          std::vector<vk::LayerProperties>    m_vk_instance_layer_properties;
          PFN_vkGetInstanceProcAddr           m_vkGetInstanceProcAddr;
          PFN_vkDestroyInstance               m_vkDestroyInstance;
#ifndef NDEBUG
          PFN_vkDestroyDebugUtilsMessengerEXT m_vkDestroyDebugUtilsMessengerEXT;
#endif
        };
        struct GFXVulkanWindow;
        struct GFXVulkanInstance : protected GFXInstanceImpl<GFXVulkanInstanceObject> {
          using impl_type = GFXInstanceImpl<GFXVulkanInstanceObject>;
          using type = typename impl_type::type;
          GFXVulkanInstance() noexcept :impl_type(type::create()) {}
          GFXVulkanInstance(const GFXVulkanInstance& v) noexcept : impl_type(v.getObject()) {}
          GFXVulkanInstance(nullptr_t) noexcept :impl_type(nullptr) {}
          GFXVulkanInstance(const std::shared_ptr<type>& obj) noexcept :impl_type(obj) {}
          template<typename GFXVulkanInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXVulkanInstanceLike::type>, nullptr_t > = nullptr>
          GFXVulkanInstance(const GFXVulkanInstanceLike& v) noexcept :impl_type(v.getObject()) {}
          GFXVulkanInstance& operator=(const GFXVulkanInstance& v) noexcept { if (this != &v) { setObject(v.getObject()); }return*this; }
          GFXVulkanInstance& operator=(const std::shared_ptr<type>& obj) noexcept { setObject(obj); return*this; }
          GFXVulkanInstance& operator=(nullptr_t) noexcept { setObject(nullptr); return*this; }
          template<typename GFXVulkanInstanceLike, std::enable_if_t<std::is_base_of_v<type, typename GFXVulkanInstanceLike::type>, nullptr_t > = nullptr>
          GFXVulkanInstance& operator=(const GFXVulkanInstanceLike& v) noexcept { setObject(v.getObject()); return*this; }
          auto createWindow(const GFXWindowDesc& desc) -> GFXVulkanWindow;
          using impl_type::operator!;
          using impl_type::operator bool;
          using impl_type::getObject;
        };
        template<typename VulkanFunctionT>
        inline auto GFXVulkanInstanceObject::getVKInstanceProcAddr(const char* name) const -> VulkanFunctionT {
          if (!m_vkGetInstanceProcAddr) { return nullptr; }
          return reinterpret_cast<VulkanFunctionT>(m_vkGetInstanceProcAddr(m_vk_instance, name));
        }
}
    }
  }
}
