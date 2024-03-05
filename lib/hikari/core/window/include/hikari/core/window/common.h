#pragma once
#include <cstdint>
#include <string>
#include <hikari/core/input/input.h>
namespace hikari {
  namespace core {
    struct WindowExtent2D {
      uint32_t width;
      uint32_t height;
    };
    struct WindowOffset2D {
      int32_t x;
      int32_t y;
    };

    enum class WindowFlagBits : U8 {
      eNone = 0x00,
      eVisible = 0x01,
      eResizable = 0x02,
      eFloating = 0x04,
      eFullscreen = 0x08,
      eGraphicsOpenGL = 0x10,
      eGraphicsVulkan = 0x20,
      eMask = 0x3F
    };
    using WindowFlags = Flags<WindowFlagBits>;
    struct WindowDesc {
      WindowExtent2D size     = {};
      WindowOffset2D position = {};
      std::string    title    = "";
      WindowFlags    flags    = WindowFlagBits::eNone;
    };
    struct Window;
    struct WindowData {
      Window*        window       = nullptr;
      WindowExtent2D size         = {};
      WindowExtent2D surface_size = {};
      WindowExtent2D fullscreen_size = {};
      WindowOffset2D position     = {};
      Input          input        = {};
      std::string    title        = "";
      bool is_closed     = false;
      bool is_focused    = false;
      bool is_hovered    = false;
      bool is_floating   = false;
      bool is_iconified  = false;
      bool is_fullscreen = false;
      bool is_resizable  = false;
      bool is_visible    = false;
      bool is_graphics_opengl = false;
      bool is_graphics_vulkan = false;
    };
  }
}
