
#include <hikari/platform/vulkan/render/gfx_vulkan_instance.h>
#include <hikari/render/gfx_window.h>
#include <hikari/platform/glfw/glfw_window_manager.h>
#include <hikari/platform/vulkan/render/gfx_vulkan_window.h>
#include <spdlog/spdlog.h>

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugLogCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* pUserData) {
  auto ms = static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity);
  auto mt = static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes);
  auto mt_str = std::string("");
  if (mt == vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) {
    mt_str = "Validation";
  }
  if (mt == vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral) {
    mt_str = "General";
  }
  if (mt == vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance) {
    mt_str = "Performance";
  }
  if (mt == vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding) {
    mt_str = "Device Address Binding";
  }
  if (ms & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
    spdlog::set_level(spdlog::level::err);
    spdlog::error("Vulkan {0} Error: {1}", mt_str.data(), pCallbackData->pMessage);
  }
  if (ms & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    spdlog::set_level(spdlog::level::warn);
    spdlog::warn("Vulkan {0} Warning: {1}", mt_str.data(), pCallbackData->pMessage);
  }
  if (ms & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Vulkan {0} Info: {1}", mt_str.data(), pCallbackData->pMessage);
  }
  if (ms & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Vulkan {0} Verbose: {1}", mt_str.data(), pCallbackData->pMessage);
  }
  return VK_FALSE;
}

hikari::platforms::vulkan::render::GFXVulkanInstanceObject::GFXVulkanInstanceObject() noexcept :
  hikari::render::GFXInstanceObject(),
  m_dll{ new vk::DynamicLoader() },
  m_vk_instance{ nullptr },
  m_vk_instance_version{ 0u },
  m_vk_instance_exten_properties{},
  m_vk_instance_layer_properties{},
  m_vkGetInstanceProcAddr{ nullptr },
  m_vkDestroyInstance{ nullptr }
#ifndef NDEBUG
  , m_vk_debug_utils_messenger{ nullptr },
  m_vkDestroyDebugUtilsMessengerEXT{ nullptr }
#endif
{
  m_vkGetInstanceProcAddr = m_dll->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
}

auto hikari::platforms::vulkan::render::GFXVulkanInstanceObject::create() -> std::shared_ptr<GFXVulkanInstanceObject> {
  auto res = std::shared_ptr<GFXVulkanInstanceObject>(new GFXVulkanInstanceObject());
  if (!res) { return nullptr; }
  if (!res->initVulkanInstance()) { return nullptr; }
  return res;
}

hikari::platforms::vulkan::render::GFXVulkanInstanceObject::~GFXVulkanInstanceObject() noexcept {
  freeVulkanInstance();
}

auto hikari::platforms::vulkan::render::GFXVulkanInstanceObject::createVulkanWindow(const GFXWindowDesc& desc) -> std::shared_ptr<GFXVulkanWindowObject> {
  return GFXVulkanWindowObject::create(shared_from_this(), desc);
}

auto hikari::platforms::vulkan::render::GFXVulkanInstanceObject::createWindow(const GFXWindowDesc& desc) -> std::shared_ptr<hikari::render::GFXWindowObject>  {
  return std::static_pointer_cast<hikari::render::GFXWindowObject>(createVulkanWindow(desc));
}

auto hikari::platforms::vulkan::render::GFXVulkanInstanceObject::getVKInstance() const -> VkInstance { return m_vk_instance; }

#ifndef NDEBUG
auto hikari::platforms::vulkan::render::GFXVulkanInstanceObject::getVKDebugUtilsMessenger() const -> VkDebugUtilsMessengerEXT { return m_vk_debug_utils_messenger; }
#endif

bool hikari::platforms::vulkan::render::GFXVulkanInstanceObject::initVulkanInstance() {
  auto& manager = glfw::WindowManager::getInstance();
  auto vkCreateInstance = getVKInstanceProcAddr<PFN_vkCreateInstance>("vkCreateInstance");
  auto vkEnumerateInstanceVersion = getVKInstanceProcAddr<PFN_vkEnumerateInstanceVersion>("vkEnumerateInstanceVersion");
  auto vkEnumerateInstanceExtensionProperties = getVKInstanceProcAddr<PFN_vkEnumerateInstanceExtensionProperties>("vkEnumerateInstanceExtensionProperties");
  auto vkEnumerateInstanceLayerProperties = getVKInstanceProcAddr<PFN_vkEnumerateInstanceLayerProperties>("vkEnumerateInstanceLayerProperties");
  auto instance_version = static_cast<uint32_t>(0);
  {
    if (!vkEnumerateInstanceVersion) { return false; }
    if (vkEnumerateInstanceVersion(&instance_version) != VK_SUCCESS) { return false; }
  }
  auto instance_extension_props = std::vector<vk::ExtensionProperties>();
  {
    auto instance_extension_count = static_cast<uint32_t>(0);
    if (!vkEnumerateInstanceExtensionProperties) { return false; }
    if (vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr) != VK_SUCCESS) { return false; }
    instance_extension_props.resize(instance_extension_count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, reinterpret_cast<VkExtensionProperties*>(instance_extension_props.data())) != VK_SUCCESS) { return false; }
  }
  auto instance_layer_props = std::vector<vk::LayerProperties>();
  {
    auto instance_layer_count = static_cast<uint32_t>(0);
    if (!vkEnumerateInstanceLayerProperties) { return false; }
    if (vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr) != VK_SUCCESS) { return false; }
    instance_layer_props.resize(instance_layer_count);
    if (vkEnumerateInstanceLayerProperties(&instance_layer_count, reinterpret_cast<VkLayerProperties*>(instance_layer_props.data())) != VK_SUCCESS) { return false; }
  }
  auto application_info = vk::ApplicationInfo();
  application_info.apiVersion = instance_version;
  application_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
  application_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
  application_info.pApplicationName = "HIKARI";
  application_info.pEngineName = "HIKARI";
  std::vector<const char*> instance_exten_names = {};
  std::vector<const char*> instance_layer_names = {};
  {
    auto required_extension_names = std::vector<const char*>{ "VK_KHR_surface", "VK_KHR_win32_surface" };
    for (auto& ext : required_extension_names) {
      auto iter = std::find_if(
        std::begin(instance_extension_props), std::end(instance_extension_props),
        [ext](const vk::ExtensionProperties& prop) { return std::string_view(prop.extensionName.data()) == ext; }
      );
      if (iter == std::end(instance_extension_props)) { return false; }
      instance_extension_props.push_back(*iter);
      instance_exten_names.push_back(ext);
    }
  }
  {
    auto iter = std::find_if(
      std::begin(instance_extension_props), std::end(instance_extension_props),
      [](const vk::ExtensionProperties& prop) { return std::string_view(prop.extensionName.data()) == VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME; }
    );
    if (iter == std::end(instance_extension_props)) { return false; }
    instance_extension_props.push_back(*iter);
    instance_exten_names.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }
#ifndef NDEBUG
  {
    auto iter = std::find_if(
      std::begin(instance_extension_props), std::end(instance_extension_props),
      [](const vk::ExtensionProperties& prop) { return std::string_view(prop.extensionName.data()) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME; }
    );
    if (iter == std::end(instance_extension_props)) { return false; }
    instance_extension_props.push_back(*iter);
    instance_exten_names.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  {
    auto iter = std::find_if(
      std::begin(instance_layer_props), std::end(instance_layer_props),
      [](const vk::LayerProperties& prop) {
        return std::string_view(prop.layerName.data()) == "VK_LAYER_KHRONOS_validation";
      }
    );
    if (iter == std::end(instance_layer_props)) { return false; }
    instance_layer_props.push_back(*iter);
    instance_layer_names.push_back("VK_LAYER_KHRONOS_validation");
  }
#endif
  auto instance_create_info = vk::InstanceCreateInfo();
  instance_create_info.pApplicationInfo = &application_info;
  instance_create_info.setPEnabledExtensionNames(instance_exten_names);
  instance_create_info.setPEnabledLayerNames(instance_layer_names);
  auto debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT();
  {
    debug_create_info.messageSeverity =
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose;

    debug_create_info.messageType =
      //vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding;

    debug_create_info.pfnUserCallback = DebugLogCallback;
#ifndef NDEBUG
    instance_create_info.pNext = &debug_create_info;
#endif
  }
  VkInstance  vk_instance = nullptr;
  if (vkCreateInstance(reinterpret_cast<VkInstanceCreateInfo*>(&instance_create_info), nullptr, &vk_instance) != VK_SUCCESS) {
    return false;
  }
  m_vk_instance = vk_instance;
  auto vkDestroyInstance = getVKInstanceProcAddr<PFN_vkDestroyInstance>("vkDestroyInstance");
#ifndef NDEBUG
  VkDebugUtilsMessengerEXT  vk_debug_utils_messenger = nullptr;
  auto vkCreateDebugUtilsMessengerEXT = getVKInstanceProcAddr<PFN_vkCreateDebugUtilsMessengerEXT>("vkCreateDebugUtilsMessengerEXT");
  if (vkCreateDebugUtilsMessengerEXT(vk_instance,reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&debug_create_info),nullptr,&vk_debug_utils_messenger)!=VK_SUCCESS) {
    vkDestroyInstance(vk_instance, nullptr);
    m_vk_instance = nullptr;
    return false;
  }
#endif
  m_vkDestroyInstance = vkDestroyInstance;
#ifndef NDEBUG
  m_vk_debug_utils_messenger = vk_debug_utils_messenger;
  m_vkDestroyDebugUtilsMessengerEXT = getVKInstanceProcAddr<PFN_vkDestroyDebugUtilsMessengerEXT>("vkDestroyDebugUtilsMessengerEXT");
#endif
  m_vk_instance_version = instance_version;
  m_vk_instance_exten_properties = instance_extension_props;
  m_vk_instance_layer_properties = instance_layer_props;
  return true;
}

void hikari::platforms::vulkan::render::GFXVulkanInstanceObject::freeVulkanInstance() {
#ifndef NDEBUG
  if (m_vk_instance && m_vkDestroyDebugUtilsMessengerEXT) {
    m_vkDestroyDebugUtilsMessengerEXT(m_vk_instance, m_vk_debug_utils_messenger,nullptr);
  }
#endif
  if (m_vk_instance && m_vkDestroyInstance) {
    m_vkDestroyInstance(m_vk_instance, nullptr);
  }
}

auto hikari::platforms::vulkan::render::GFXVulkanInstance::createWindow(const GFXWindowDesc& desc) -> GFXVulkanWindow {
  auto obj = getObject();
  if (!obj) { return GFXVulkanWindow(nullptr); }
  return GFXVulkanWindow(obj->createVulkanWindow(desc));
}
