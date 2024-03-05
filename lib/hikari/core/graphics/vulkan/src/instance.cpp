#include <hikari/core/graphics/vulkan/instance.h>
bool hikari::core::GraphicsVulkanInstance::create() {
  if (*m_instance) { return true; }
  auto ext_names = vk_utils::toNames(m_extension_properties);
  auto lyr_names = vk_utils::toNames(m_layer_properties);

  vk::ApplicationInfo app_info;
  app_info.setApiVersion(m_api_version);
  app_info.setPApplicationName("hikari");
  app_info.setPEngineName("hikari");

  vk::InstanceCreateInfo inst_info;
  inst_info.setPEnabledExtensionNames(ext_names);
  inst_info.setPEnabledLayerNames(lyr_names);
  inst_info.setPApplicationInfo(&app_info);

  auto inst = hikari::core::GraphicsVulkanContext::getInstance().createInstance(inst_info);
  m_instance = std::move(inst);
  m_physical_devices = vk::raii::PhysicalDevices(m_instance);
  m_physical_device_infos.reserve(m_physical_devices.size());
  for (auto& physical_device : m_physical_devices) {
    auto info = GraphicsVulkanPhysicalDeviceInfo();
    info.init(physical_device);
    m_physical_device_infos.push_back(std::move(info));
  }

  return true;
}
void hikari::core::GraphicsVulkanInstance::release() noexcept {
  m_physical_devices = nullptr;
  m_instance = nullptr;
}
