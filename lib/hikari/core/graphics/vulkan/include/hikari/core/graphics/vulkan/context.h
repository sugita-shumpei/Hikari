#pragma once
#include <memory>
#include <vector>
#include <hikari/core/graphics/vulkan/common.h>
namespace hikari {
  namespace core {
    struct GraphicsVulkanContext {
      static auto getInstance() -> GraphicsVulkanContext& {
        static GraphicsVulkanContext context;
        return context;
      }
      ~GraphicsVulkanContext() noexcept {}
      auto enumerateInstanceExtensionProperties() const noexcept -> const  std::vector<vk::ExtensionProperties>& { return m_extension_properties; }
      auto enumerateInstanceLayerProperties() const noexcept -> const  std::vector<vk::LayerProperties>& { return m_layer_properties; }
      auto enumerateInstanceVersion() const noexcept -> uint32_t { return m_api_version; }
      auto createInstance(const vk::InstanceCreateInfo& info) const -> vk::raii::Instance {
        return m_handle.createInstance(info);
      }
    private:
      GraphicsVulkanContext()  noexcept : m_handle{}, m_extension_properties{}, m_layer_properties{}, m_api_version{} {
        m_extension_properties = m_handle.enumerateInstanceExtensionProperties();
        m_layer_properties = m_handle.enumerateInstanceLayerProperties();
        m_api_version = m_handle.enumerateInstanceVersion();
      }
      GraphicsVulkanContext(const GraphicsVulkanContext&) noexcept = delete;
      GraphicsVulkanContext(GraphicsVulkanContext&&) noexcept = delete;
      GraphicsVulkanContext& operator=(const GraphicsVulkanContext&) noexcept = delete;
      GraphicsVulkanContext& operator=(GraphicsVulkanContext&&) noexcept = delete;
      vk::raii::Context m_handle;
      std::vector<vk::ExtensionProperties> m_extension_properties;
      std::vector<vk::LayerProperties> m_layer_properties;
      uint32_t m_api_version;
    };
  }
}
