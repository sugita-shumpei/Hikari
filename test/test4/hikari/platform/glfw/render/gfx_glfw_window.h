#pragma once
#include <hikari/render/gfx_window.h>
#include <hikari/platform/opengl/render/gfx_opengl_instance.h>
struct GLFWwindow;
namespace hikari {
  inline namespace platforms {
    namespace glfw {
      inline namespace render {
        struct GFXGLFWWindowObject : public hikari::render::GFXWindowObject {
          ~GFXGLFWWindowObject() noexcept;
          auto getHandle() const -> void* override;
          bool isResizable() const override;
          bool isClosed() const override;
          bool isFocused() const override;
          bool isVisible() const override;
          bool isIconified() const override;
          bool isBorderless() const override;
          void setResizable(bool resizable) override;
          void setVisible(bool visible) override;
          void setIconified(bool iconified) override;
          void setBorderless(bool borderless) override;
          auto getClipboard() const->std::string override;
          void setClipboard(const std::string& s) override;
          auto getTitle() const->std::string override;
          void setTitle(const std::string& s) override;
          auto getSize() const->std::array<uint32_t, 2> override;
          auto getPosition() const->std::array<uint32_t, 2> override;
          auto getFramebufferSize() const->std::array<uint32_t, 2> override;
          void setSize(const std::array<uint32_t, 2>& size) override;
          void setPosition(const std::array<uint32_t, 2>& pos) override;
        protected:
          GFXGLFWWindowObject(GLFWwindow* window, const GFXWindowDesc& desc) noexcept;
          void initEvents();
        private:
          bool m_close = false;
          bool m_visible = false;
          bool m_focus = false;
          bool m_resizable = false;
          bool m_iconified  = false;
          bool m_borderless = false;
          GLFWwindow* m_window = nullptr;
          std::string m_title = "";
          std::array<uint32_t, 2> m_size = {};
          std::array<uint32_t, 2> m_fb_size = {};
          std::array<uint32_t, 2> m_position = {};
          std::array<double, 2> m_cursor_position = {};
          std::array<double, 2> m_scroll = {};

#define HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(NAME,...) \
  using PFN_##NAME = void(*)(GLFWwindow* handle,__VA_ARGS__); \
  static void Default##NAME(GLFWwindow* handle,__VA_ARGS__)
#define HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK_VOID(NAME) \
  using PFN_##NAME = void(*)(GLFWwindow* handle); \
  static void Default##NAME(GLFWwindow* handle)

          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowSize, int32_t w, int32_t h);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowPosition, int32_t x, int32_t y);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK_VOID(CallbackWindowClose);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackWindowIconified, int32_t iconified);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackFramebufferSize, int32_t w, int32_t h);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackCursorPosition, double x, double y);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackKey, int32_t key, int32_t scancode, int32_t action, int32_t mods);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackChar, uint32_t codepoint);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackMouseButton, int32_t button, int32_t action, int32_t mods);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackCursorEnter, int32_t enter);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackDrop, int path_count, const char* paths[]);
          HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK(CallbackScroll, double x, double y);

#undef HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK
#undef HK_PLATFORM_GLFW_DEFINE_NATIVE_CALLBACK_VOID
        };
      }
    }
  }
}
