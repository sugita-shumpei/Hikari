#pragma once
#include <hikari/core/data_type.h>
struct GLFWwindow;
namespace hikari {
  inline namespace platforms {
    namespace glfw {
      using gl_proc = void(*)(void);
      struct WindowManager {
        static auto getInstance() -> WindowManager&;
        ~WindowManager()noexcept;
        auto createOpenGLWindow(int w, int h, int x, int y, const char* title, bool is_visible, bool is_resizable, bool is_borderless, void* share_ctx = nullptr) -> void*;
        auto createVulkanWindow(int w, int h, int x, int y, const char* title, bool is_visible, bool is_resizable, bool is_borderless) -> void*;
        auto createVulkanSurface(void* window, void* vk_instance) -> void*;
        bool supportVulkan() const;
        auto getVulkanSurfaceExtensionNames() const -> std::vector<const char*>;
        void destroyWindow(void* window);
        void setCurrentContext(void* ctx);
        auto getCurrentContext() -> void*;
        void pollEvents();
        void swapBuffers(void* window);
        bool shouldCloseWindow(void* window);
        void showWindow(void* window);
        void hideWindow(void* window);
        auto getWindowUserPointer(void* window) -> void*;
        void setWindowUserPointer(void* window, void*);
        auto getClipboard(void* window) -> const char*;
        void setClipboard(void* window, const char* name);
        void setWindowTitle(void* window, const char* name);
        void getWindowSize(void* window, int&w, int&h);
        void setWindowSize(void* window, int w, int h);
        void setWindowPosition(void* window, int x, int y);
        void getWindowPosition(void* window, int&x, int&y);
        void getFramebufferSize(void* window, int& w, int& h);
        void setWindowResizable(void* window, bool);
        void setWindowVisible(void* window, bool);
        void setWindowIconified(void* window);
        void setWindowBorderless(void* window, bool);
        static gl_proc(*GetProcAddress)(const char* name);

#define HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(NAME, ...) \
  void set##NAME(void* window, void(*)(GLFWwindow* handle,__VA_ARGS__))
#define HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK_VOID(NAME) \
  void set##NAME(void* window, void(*)(GLFWwindow* handle))

        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowSize, int32_t w, int32_t h);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowPosition, int32_t x, int32_t y);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK_VOID(CallbackWindowClose);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowIconified, int32_t iconified);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackFramebufferSize, int32_t w, int32_t h);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackCursorPosition, double x, double y);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackKey, int32_t key, int32_t scancode, int32_t action, int32_t mods);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackChar, uint32_t codepoint);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackMouseButton, int32_t button, int32_t action, int32_t mods);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackCursorEnter, int32_t enter);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackDrop, int path_count, const char* paths[]);
        HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackScroll, double x, double y);

#undef HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK
#undef HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK_VOID
      private:
        WindowManager() noexcept;
        WindowManager(const WindowManager&) = delete;
        WindowManager(WindowManager&&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;
        WindowManager& operator=(WindowManager&&) = delete;
      };
    }
  }
}
