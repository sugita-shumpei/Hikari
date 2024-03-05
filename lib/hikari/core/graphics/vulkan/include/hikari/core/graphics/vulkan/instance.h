#pragma once
#include <hikari/core/graphics/vulkan/context.h>
#include <hikari/core/graphics/vulkan/common.h>
namespace hikari {
  namespace core {
    struct GraphicsVulkanPhysicalDeviceInfo {
      void init(const vk::raii::PhysicalDevice& physical_device) {
        features = physical_device.getFeatures();
        properties = physical_device.getProperties();
        extension_properties = physical_device.enumerateDeviceExtensionProperties();
        queue_family_properties = physical_device.getQueueFamilyProperties();
      }
      bool supportApiVersion(uint32_t version)const noexcept {
        return properties.apiVersion >= version;
      }
      bool supportExtension(const char* name)const noexcept { return vk_utils::findName(extension_properties, name) != std::end(extension_properties); }
      vk::PhysicalDeviceFeatures features = {};
      vk::PhysicalDeviceProperties properties = {};
      std::vector<vk::ExtensionProperties> extension_properties = {};
      std::vector<vk::QueueFamilyProperties> queue_family_properties = {};
    };
    struct GraphicsVulkanInstance {
      GraphicsVulkanInstance() noexcept :m_instance(nullptr), m_physical_devices(nullptr), m_api_version{ VK_API_VERSION_1_0 }, m_extension_properties{}, m_layer_properties{}, m_physical_device_infos{} {}
      ~GraphicsVulkanInstance() noexcept { release(); }

      GraphicsVulkanInstance(nullptr_t)noexcept :GraphicsVulkanInstance() {}
      GraphicsVulkanInstance(const GraphicsVulkanInstance& lhs) noexcept = delete;
      GraphicsVulkanInstance(GraphicsVulkanInstance&& rhs) noexcept = default;
      GraphicsVulkanInstance& operator=(const GraphicsVulkanInstance& lhs) noexcept = delete;
      GraphicsVulkanInstance& operator=(GraphicsVulkanInstance&& rhs) noexcept = default;
      GraphicsVulkanInstance& operator=(nullptr_t) noexcept {
        release();
        return *this;
      }

      explicit operator vk::raii::Instance& () noexcept { return m_instance; }

      auto operator->() const noexcept -> const vk::raii::Instance* {
        return &m_instance;
      }
      auto operator->() noexcept -> vk::raii::Instance* {
        return &m_instance;
      }
      auto operator* () const noexcept -> vk::Instance { return *m_instance; }

      bool create();
      void release() noexcept;

      auto getEnabledApiVersion() const -> std::uint32_t { return m_api_version; }
      auto getEnabledExtensionProperties() const -> const std::vector<vk::ExtensionProperties>& { return m_extension_properties; }
      auto getEnabledLayerProperties() const -> const std::vector<vk::LayerProperties>& { return m_layer_properties; }
      bool hasEnabledExtension(const char* name)const noexcept { return vk_utils::findName(m_extension_properties, name) != std::end(m_extension_properties); }
      bool hasEnabledLayer(const char* name)const noexcept { return vk_utils::findName(m_layer_properties, name) != std::end(m_layer_properties); }

      auto getSupportedApiVersion() const-> std::uint32_t {
        return GraphicsVulkanContext::getInstance().enumerateInstanceVersion();
      }
      auto getSupportedExtensionProperties() const -> const std::vector<vk::ExtensionProperties>& {
        return GraphicsVulkanContext::getInstance().enumerateInstanceExtensionProperties();
      }
      auto getSupportedLayerProperties() const -> const std::vector<vk::LayerProperties>& {
        return GraphicsVulkanContext::getInstance().enumerateInstanceLayerProperties();
      }
      auto getSupportedPhysicalDevices() const noexcept -> const vk::raii::PhysicalDevices& { return m_physical_devices; }
      auto getSupportedPhysicalDeviceCount() const noexcept -> uint32_t { return m_physical_devices.size(); }
      auto getSupportedPhysicalDeviceInfo(uint32_t index) const -> const GraphicsVulkanPhysicalDeviceInfo& { return m_physical_device_infos.at(index); }

      bool requestApiVersion(uint32_t version) noexcept {
        if (*m_instance) { return false; }
        if (getSupportedApiVersion() >= version) {
          m_api_version = version;
          return true;
        }
        return false;
      }
      bool requestExtension(const char* name) noexcept
      {
        if (vk_utils::findName(m_extension_properties, name) != std::end(m_extension_properties)) {
          return true;
        }
        if (*m_instance) { return false; }
        auto& props = getSupportedExtensionProperties();
        auto iter = vk_utils::findName(props, name);
        if (iter != std::end(props)) {
          m_extension_properties.push_back(*iter);
          return true;
        }
        return false;
      }
      bool requestLayer(const char* name) noexcept
      {
        if (vk_utils::findName(m_layer_properties, name) != std::end(m_layer_properties)) {
          return true;
        }
        if (*m_instance) { return false; }
        auto& props = getSupportedLayerProperties();
        auto iter = vk_utils::findName(props, name);
        if (iter != std::end(props)) {
          m_layer_properties.push_back(*iter);
          return true;
        }
        return false;
      }

      auto getVkInstance() const noexcept -> vk::Instance { return *m_instance; }
    private:
      vk::raii::Instance m_instance;
      vk::raii::PhysicalDevices m_physical_devices;
      std::uint32_t m_api_version;
      std::vector<vk::ExtensionProperties> m_extension_properties;
      std::vector<vk::LayerProperties> m_layer_properties;
      std::vector<GraphicsVulkanPhysicalDeviceInfo> m_physical_device_infos;
    };
  }
}
