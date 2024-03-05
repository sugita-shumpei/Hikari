#include <hikari/core/graphics/vulkan/device.h>

bool hikari::core::GraphicsVulkanDevice::create() {
  if (*m_device) { return true; }
  if (!requestGeneralQueue()) { return false; }
  auto device_queue_create_infos = std::vector<vk::DeviceQueueCreateInfo>();
  {
    device_queue_create_infos.reserve(m_device_queue_infos.size());
    for (auto& info : m_device_queue_infos) {
      device_queue_create_infos.push_back(info.toCreateInfo());
    }
  }
  auto    extension_names  = vk_utils::toNames(m_enabled_info.extension_properties);
  auto create_info = vk::DeviceCreateInfo()
    .setPEnabledExtensionNames(extension_names)
    .setQueueCreateInfos(device_queue_create_infos);

  auto    features = m_enabled_info.features;
  auto   features2 = vk::PhysicalDeviceFeatures2(features);
  auto features_chain = VulkanPNextChain();
  void* p_next = nullptr;
  if (m_enabled_features2.getCount() != 0) {
    features_chain  = VulkanPNextChain(m_enabled_features2);
    features2.pNext = features_chain.getPHead();
    create_info.pNext = &features2;
  }
  else {
    create_info.pEnabledFeatures = &features;
  }


  m_device = getEnabledPhysicalDevice().createDevice(create_info);
  {
    for (auto& device_queue_info : m_device_queue_infos) {
      auto i = 0u;
      for (auto& priority : device_queue_info.priorities) {
        device_queue_info.queues.emplace_back(m_device.getQueue(device_queue_info.family_index, i));
        ++i;
      }
    }
  }
  return true;
}

void hikari::core::GraphicsVulkanDevice::release() noexcept {
  {
    for (auto& device_queue_info : m_device_queue_infos) {
      device_queue_info.queues.clear();
    }
  }
  m_device = nullptr;
}

void hikari::core::GraphicsVulkanDevice::supportFeatures(vk::PhysicalDeviceFeatures& feat) const
{
  feat = getEnabledPhysicalDevice().getFeatures();
}

void hikari::core::GraphicsVulkanDevice::supportFeatures2(VulkanPNextChain& chain) const
{
  auto vkGetPhysicalDeviceFeatures2 = m_instance->getDispatcher()->vkGetPhysicalDeviceFeatures2;
  if (vkGetPhysicalDeviceFeatures2) {
    vk::PhysicalDeviceFeatures2 features = {};
    features.pNext = chain.getPHead();
    vkGetPhysicalDeviceFeatures2(getVkPhysicalDevice(), reinterpret_cast<VkPhysicalDeviceFeatures2*>(&features));
    return;
  }
  auto vkGetPhysicalDeviceFeatures2KHR = m_instance->getDispatcher()->vkGetPhysicalDeviceFeatures2KHR;
  if (vkGetPhysicalDeviceFeatures2KHR) {
    vk::PhysicalDeviceFeatures2KHR features = {};
    features.pNext = chain.getPHead();
    vkGetPhysicalDeviceFeatures2KHR(getVkPhysicalDevice(), reinterpret_cast<VkPhysicalDeviceFeatures2KHR*>(&features));
    return;
  }
}

bool hikari::core::GraphicsVulkanDevice::supportQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits) const noexcept
{
  auto& props = getSupportedQueueFamilyProperties();
  return vk_utils::findQueueFamily(props,required_bits,avoided_bits) != std::end(props);
}

bool hikari::core::GraphicsVulkanDevice::supportQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits, uint32_t& count) const noexcept
{
  auto& props = getSupportedQueueFamilyProperties();
  auto iter= vk_utils::findQueueFamily(props, required_bits, avoided_bits);
  if (iter == std::end(props)) { return false; }
  count = iter->queueCount;
  return true;
}

bool hikari::core::GraphicsVulkanDevice::requestFeatures(fu2::unique_function<bool(const GraphicsVulkanDevice&, vk::PhysicalDeviceFeatures& features)>&& cull_callback)
{
  if (*m_device) { return false; }
  vk::PhysicalDeviceFeatures features;
  supportFeatures(features);
  if (!cull_callback(*this, features)) { return false; }
  m_enabled_info.features = features;
  return true;
}

bool hikari::core::GraphicsVulkanDevice::requestFeatures2(VulkanPNextChain& chain,
  fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain&)>&& cull_callback
)
{
  if (*m_device) { return false; }
  if (chain.getCount() == 0) { return false; }
  supportFeatures2(chain);
  if (!cull_callback(*this, chain)) { return false; }
  m_enabled_features2.setChain(chain);
  return true;
}

bool hikari::core::GraphicsVulkanDevice::requestFeatures2(

  fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain&)>&& init_callback,
  fu2::unique_function<bool(const GraphicsVulkanDevice&, VulkanPNextChain&)>&& cull_callback
)
{
  if (*m_device) { return false; }
  VulkanPNextChain chain;
  if (!init_callback(*this, chain)) { return false; }
  return requestFeatures2(chain,std::move(cull_callback));
}

bool hikari::core::GraphicsVulkanDevice::requestQueueFamily(vk::QueueFlags required_bits, vk::QueueFlags avoided_bits,const std::vector<float>& priorities) noexcept
{
  uint32_t queue_count = 0;
  auto& props = getSupportedQueueFamilyProperties();
  auto iter = vk_utils::findQueueFamily(props, required_bits, avoided_bits);
  if (iter == std::end(props)) {
    return false;
  }
  if (iter->queueCount < priorities.size()) {
    return false;
  }
  uint32_t family_index = std::distance(std::begin(props), iter);
  auto tmp = std::ranges::find_if(m_device_queue_infos, [family_index](const auto& info) {
    return info.family_index == family_index;
    });
  if (tmp == std::end(m_device_queue_infos)) {
    m_device_queue_infos.push_back(GraphicsVulkanDeviceQueueInfo{ family_index,iter->queueFlags,priorities });
  }
  else {
    tmp->priorities = priorities;
  }
  m_enabled_info.queue_family_properties[family_index].queueCount = priorities.size();
  return true;
}

bool hikari::core::GraphicsVulkanDevice::requestGeneralQueue()
{
  if (*m_device) { return false; }
  {
    auto props = getEnabledQueueFamilyProperties();
    auto iter = vk_utils::findQueueFamily(props, vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eCompute, {});
    if (iter != std::end(props)) {
      if (iter->queueCount > 0) { return true; }
    }
  }
  return requestQueueFamily(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eCompute, {}, std::vector<float>{ 1.0f });
}

