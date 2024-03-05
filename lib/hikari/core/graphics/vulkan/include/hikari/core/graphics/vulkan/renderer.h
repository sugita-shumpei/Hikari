#pragma once
#include <hikari/core/graphics/vulkan/device.h>
#include <hikari/core/window/renderer.h>
#include <hikari/core/window/window.h>
namespace hikari{
  namespace core {
    struct GraphicsVulkanRenderer : public hikari::core::WindowRenderer {
      GraphicsVulkanRenderer(Window* window);
      virtual ~GraphicsVulkanRenderer() noexcept;
      bool initialize() override final;
      void terminate() override final;
      auto getInstance() -> GraphicsVulkanInstance&;
      auto getDevice() -> GraphicsVulkanDevice&;
    protected:
      void setInstance(std::unique_ptr<GraphicsVulkanInstance>&& instance);
      void setDevice(std::unique_ptr<GraphicsVulkanDevice>&& device);
    private:
      virtual bool initInstance();
      bool initSurface();
      virtual bool initDevice();
      virtual bool initCustom();
      void freeInstance();
      void freeSurface();
      void freeDevice();
      virtual void freeCustom();
    private:
      std::unique_ptr<GraphicsVulkanInstance> m_instance = nullptr;
      std::unique_ptr<vk::raii::SurfaceKHR>   m_surface  = nullptr;
      std::unique_ptr<GraphicsVulkanDevice>   m_device   = nullptr;
      bool m_is_init = false;
      bool m_is_init_instance = false;
      bool m_is_init_surface = false;
      bool m_is_init_device = false;
      bool m_is_init_extra = false;
    };
  };
}
