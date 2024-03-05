#pragma once
#include <function2/function2.hpp>
#include <hikari/core/graphics/vulkan/instance.h>
namespace hikari {
  namespace core {
    struct GraphicsVulkanDeviceQueueInfo {
      std::uint32_t      family_index = {};
      vk::QueueFlags     flags        = {};
      std::vector<float> priorities   = {};
      std::vector<vk::raii::Queue> queues = {};
      auto toCreateInfo() const noexcept -> vk::DeviceQueueCreateInfo {
        return vk::DeviceQueueCreateInfo().setQueueFamilyIndex(family_index).setQueuePriorities(priorities);
      }
    };
    struct GraphicsVulkanDevice {
      GraphicsVulkanDevice(GraphicsVulkanInstance& instance) :
        m_instance{ instance },
        m_device(nullptr),
        m_physical_device_index{ 0u },
        m_enabled_info{},
        m_enabled_features2{},
        m_device_queue_infos{}
      {
        m_enabled_info.properties = getSupportedProperties();
        m_enabled_info.queue_family_properties = getSupportedQueueFamilyProperties();
        for (auto& prop : m_enabled_info.queue_family_properties) { prop.queueCount = 0; }
      }
      ~GraphicsVulkanDevice() noexcept {
        release();
      }
      GraphicsVulkanDevice(const GraphicsVulkanDevice& lhs) noexcept = delete;
      GraphicsVulkanDevice(GraphicsVulkanDevice&& rhs) noexcept = default;
      GraphicsVulkanDevice& operator=(const GraphicsVulkanDevice& lhs) noexcept = delete;
      GraphicsVulkanDevice& operator=(GraphicsVulkanDevice&& rhs) noexcept = default;
      GraphicsVulkanDevice& operator=(nullptr_t) noexcept {
        release();
        return *this;
      }

      auto operator->() const -> const vk::raii::Device& {
        return m_device;
      }
      auto operator->() -> vk::raii::Device& {
        return m_device;
      }

      bool create();
      void release() noexcept;

      auto getEnabledDeviceQueueInfos() const noexcept -> const std::vector<GraphicsVulkanDeviceQueueInfo>& { return m_device_queue_infos; }
      auto getQueue(uint32_t queue_family_index, uint32_t queue_index) const -> const vk::raii::Queue& {
        auto iter = std::ranges::find_if(m_device_queue_infos, [queue_family_index, queue_index](const auto& p) {
          if (p.family_index != queue_family_index) { return false; }
          if (p.priorities.size() <= queue_index) { return false; }
          return true;
          });
        if (iter != std::end(m_device_queue_infos)) { return iter->queues[queue_index]; }
        else {
          throw std::out_of_range("[hikari/core/graphics/vulkan/device.h line 59]");
        }
      }
      auto findQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits) const noexcept -> std::optional<uint32_t> {
        auto& props = getEnabledQueueFamilyProperties();
        auto iter = vk_utils::findQueueFamily(props, required_bits, avoided_bits);
        if (iter !=std::end(props)) {
          if (iter->queueCount == 0) { return std::nullopt; }
          return std::distance(std::begin(props), iter);
        }
        else {
          return std::nullopt;
        }
      }

      auto getEnabledPhysicalDevice() const noexcept -> const vk::raii::PhysicalDevice& { return m_instance.getSupportedPhysicalDevices()[m_physical_device_index]; }
      auto getEnabledPhysicalDeviceInfo() const noexcept -> const GraphicsVulkanPhysicalDeviceInfo& { return m_enabled_info; }
      auto getEnabledFeatures() const noexcept -> const vk::PhysicalDeviceFeatures& { return m_enabled_info.features; }
      auto getEnabledFeatures2() const noexcept -> const VulkanPNextChainBuilder& { return m_enabled_features2; }
      auto getEnabledExtensionProperties() const -> const std::vector<vk::ExtensionProperties>& { return m_enabled_info.extension_properties; }
      auto getEnabledQueueFamilyProperties() const noexcept -> const std::vector<vk::QueueFamilyProperties>& { return m_enabled_info.queue_family_properties; }
      auto getEnabledApiVersion() const-> std::uint32_t {
        return getEnabledPhysicalDeviceInfo().properties.apiVersion;
      }
      bool hasEnabledExtension(const char* name)const noexcept { return vk_utils::findName(m_enabled_info.extension_properties, name) != std::end(m_enabled_info.extension_properties); }

      auto getSupportedPhysicalDeviceCount() const noexcept -> uint32_t { return m_instance.getSupportedPhysicalDeviceCount(); }
      auto getSupportedPhysicalDevices() const noexcept -> const vk::raii::PhysicalDevices& { return m_instance.getSupportedPhysicalDevices(); }
      auto getSupportedPhysicalDeviceInfo() const -> const GraphicsVulkanPhysicalDeviceInfo& { return m_instance.getSupportedPhysicalDeviceInfo(m_physical_device_index); }
      auto getSupportedProperties() const-> const vk::PhysicalDeviceProperties& { return getSupportedPhysicalDeviceInfo().properties; }
      auto getSupportedFeatures() const-> const vk::PhysicalDeviceFeatures& { return getSupportedPhysicalDeviceInfo().features; }
      auto getSupportedExtensionProperties() const-> const std::vector<vk::ExtensionProperties>& { return getSupportedPhysicalDeviceInfo().extension_properties; }
      auto getSupportedQueueFamilyProperties() const-> const std::vector<vk::QueueFamilyProperties>& { return getSupportedPhysicalDeviceInfo().queue_family_properties; }
      auto getSupportedApiVersion() const-> std::uint32_t {
        return getSupportedPhysicalDeviceInfo().properties.apiVersion;
      }

      bool supportApiVersion(uint32_t version)const {
        return getSupportedPhysicalDeviceInfo().supportApiVersion(version);
      }
      void supportFeatures (vk::PhysicalDeviceFeatures& feat) const;
      void supportFeatures2(VulkanPNextChain&          chain) const;
      bool supportQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits) const noexcept;
      bool supportQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits, uint32_t& count) const noexcept;

      bool supportExtension(const char* name)const noexcept { 
        return getSupportedPhysicalDeviceInfo().supportExtension(name);
      }
      bool requestExtension(const char* name) noexcept
      {
        if (vk_utils::findName(m_enabled_info.extension_properties, name) != std::end(m_enabled_info.extension_properties)) {
          return true;
        }
        if (*m_device) { return false; }
        auto& props = getSupportedExtensionProperties();
        auto iter = vk_utils::findName(props, name);
        if (iter != std::end(props)) {
          m_enabled_info.extension_properties.push_back(*iter);
          return true;
        }
        return false;
      }
      bool requestFeatures (fu2::unique_function<bool(const GraphicsVulkanDevice&, vk::PhysicalDeviceFeatures& features)>&& cull_callback);
      bool requestFeatures2(VulkanPNextChain& chain,fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain& chain)>&& cull_callback);
      bool requestFeatures2(
        fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain& chain)>&& init_callback,
        fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain& chain)>&& cull_callback);
      bool requestQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits, const  std::vector<float>& priorities) noexcept;

      auto getVkPhysicalDevice() const noexcept -> vk::PhysicalDevice { return *getEnabledPhysicalDevice(); }
    private:
      bool requestGeneralQueue();
    private:
      GraphicsVulkanInstance& m_instance;
      vk::raii::Device m_device;
      GraphicsVulkanPhysicalDeviceInfo m_enabled_info;
      VulkanPNextChainBuilder m_enabled_features2;
      std::vector<GraphicsVulkanDeviceQueueInfo> m_device_queue_infos;
      uint32_t m_physical_device_index;
    };
  }
}
