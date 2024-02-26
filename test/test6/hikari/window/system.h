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
    auto createWindow(
      const std::string& title,
      U32 width, U32    height,
      I32 pos_x, I32     pos_y,
      GraphicsAPIType api_type,
      Bool is_floating = false,
      Bool is_resizable = false,
      Bool is_visible = false,
      Bool is_fullscreen = false
    ) -> Window*;
    void destroyWindow(Window* window);
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
