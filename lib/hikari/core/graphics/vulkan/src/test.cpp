#include <hikari/core/graphics/vulkan/common.h>
#include <hikari/core/graphics/vulkan/instance.h>
#include <hikari/core/graphics/vulkan/device.h>
#include <hikari/core/graphics/vulkan/context.h>
//#include <hikari/core/graphics/vulkan/renderer.h>
int main() {
  //{
  //  auto ptr = hikari::core::GraphicsVulkanRenderer(nullptr);
  //}
  hikari::core::GraphicsVulkanInstance instance;
  instance.requestApiVersion(VK_API_VERSION_1_3);
  instance.requestExtension("VK_KHR_surface");
  instance.requestExtension("VK_KHR_win32_surface");
  instance.requestExtension("VK_EXT_debug_utils");
  instance.requestLayer("VK_LAYER_KHRONOS_validation");
  // instance.requestLayer("VK_LAYER_LUNARG_api_dump");
  instance.create();

  hikari::core::GraphicsVulkanDevice device(instance);
  device.requestExtension("VK_KHR_swapchain");
  device.requestExtension("VK_KHR_ray_tracing_pipeline");
  device.requestExtension("VK_KHR_ray_query");
  device.requestExtension("VK_KHR_acceleration_structure");
  device.requestExtension("VK_KHR_pipeline_library");
  device.requestExtension("VK_KHR_deferred_host_operations");

  if (!device.requestFeatures(
    [](const hikari::core::GraphicsVulkanDevice& device_,  vk::PhysicalDeviceFeatures& features) {
      features.robustBufferAccess = VK_FALSE;
      if (!features.geometryShader) { return false; }
      if (!features.tessellationShader) { return false; }
      return true;
    })){
    throw std::runtime_error("Failed To Request Vulkan Device Core Features!");
  }

  if (!device.requestFeatures2(
    [](const hikari::core::GraphicsVulkanDevice& device_, hikari::core::VulkanPNextChain& chain) {
      auto builder = hikari::core::VulkanPNextChainBuilder();
      if (device_.getSupportedApiVersion() >= VK_API_VERSION_1_2) {
        builder.addStructs<vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features>();
      }
      if (device_.getSupportedApiVersion() >= VK_API_VERSION_1_3) {
        builder.addStructs<vk::PhysicalDeviceVulkan13Features>();
      }
      if (device_.supportExtension("VK_KHR_ray_tracing_pipeline")) {
        builder.addStruct<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
      }
      if (device_.supportExtension("VK_KHR_ray_query")) {
        builder.addStruct<vk::PhysicalDeviceRayQueryFeaturesKHR>();
      }
      if (device_.supportExtension("VK_KHR_acceleration_structure")) {
        builder.addStruct<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
      }
      chain = builder.build();
      return true;
    },
    [](const hikari::core::GraphicsVulkanDevice& device_, hikari::core::VulkanPNextChain& chain) {
      auto builder = hikari::core::VulkanPNextChainBuilder(std::move(chain));
      auto vulkan13_features = builder.getStruct<vk::PhysicalDeviceVulkan13Features>();
      if (vulkan13_features) {
        vulkan13_features->robustImageAccess = VK_FALSE;
        builder.setStruct(*vulkan13_features);
      }
      chain = builder.build();
      return true;
    }
  )) {
    throw std::runtime_error("Failed To Request Vulkan Device Extension Features!");
  }

  if (!device.requestQueueFamily(vk::QueueFlagBits::eGraphics| vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer, {}, std::vector{ 1.0f })) {}
  if (!device.requestQueueFamily(vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eGraphics, std::vector{ 1.0f })) {}
  if (!device.requestQueueFamily(vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute , std::vector{ 1.0f })) {}
  device.create();

  return 0;
}
