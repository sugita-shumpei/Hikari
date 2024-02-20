#pragma once
#include <memory>
#include <string>
#include <vector>
namespace hikari {
  inline namespace render {
    enum class GFXAPI {
      eUnknown,
      eOpenGL,
      eVulkan
    };
    struct GFXWindowDesc {
      uint32_t width  = 0;
      uint32_t height = 0;
      uint32_t x = UINT32_MAX;
      uint32_t y = UINT32_MAX;
      const char* title  = "";
      bool is_resizable  = false;
      bool is_visible    = false;
      bool is_borderless = false;
      bool is_fullscreen = false;
    };
    struct GFXDeviceDesc {
      const char*        name                           = nullptr;
      std::vector<float> priorities_for_transfer_queues = {};
      std::vector<float> priorities_for_compute_queues  = {};
    };

    template<typename ObjectT>
    struct GFXWrapper {
      using type = ObjectT;
      GFXWrapper() noexcept :m_impl{ nullptr } {}
      GFXWrapper(const std::shared_ptr<type>& obj) noexcept :m_impl{ obj } {}
      GFXWrapper(nullptr_t) noexcept :m_impl{ nullptr } {}
      operator bool()  const { return m_impl!=nullptr; }
      bool operator!() const { return!m_impl; }
      auto getObject() const ->std::shared_ptr<type> { return m_impl; }
      auto getAPI() const->GFXAPI {
        auto obj = getObject();
        if (!obj) { return GFXAPI::eUnknown; }
        else { return obj->getAPI(); }
      }
    protected:
      void setObject(const std::shared_ptr<type>& obj) { m_impl = obj; }
    private:
      std::shared_ptr<ObjectT> m_impl;
    };
  }
}
