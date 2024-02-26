#pragma once
#include <cstdint>
#include <hikari/core/data_types.h>
#include <hikari/graphics/common.h>
namespace hikari {
  struct Window;
  struct WindowSystem {
    static auto getInstance() -> WindowSystem&;
    ~WindowSystem() noexcept {
      terminate();
    }
    bool isInitialized();
    bool initialize();
    void terminate();
  private:
    WindowSystem() noexcept {}
    WindowSystem(const WindowSystem&) noexcept = delete;
    WindowSystem(WindowSystem&&) noexcept = delete;
    WindowSystem& operator=(const WindowSystem&) noexcept = delete;
    WindowSystem& operator=(WindowSystem&&) noexcept = delete;
  private:
    bool m_is_initialized = false;
  };
}
