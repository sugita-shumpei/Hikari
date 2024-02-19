#define GLFW_INCLUDE_VULKAN
#include <hikari/platform/glfw/glfw_window_manager.h>
#include <GLFW/glfw3.h>
#include "glfw_window_manager.h"

auto hikari::platforms::glfw::WindowManager::getInstance() -> WindowManager &
{
    static WindowManager manager;
    return manager;
}

hikari::platforms::glfw::WindowManager::~WindowManager() noexcept { glfwTerminate(); }

auto hikari::platforms::glfw::WindowManager::createOpenGLWindow(int w, int h, int x, int y, const char *title, bool is_visible, bool is_resizable, bool is_borderless, void *share_ctx) -> void *
{
    std::vector<std::pair<int, int>> versions = {
        {4, 6}, {4, 5}, {4, 4}, {4, 3}, {4, 2}, {4, 1}, {4, 0}, {3, 3}, {3, 2}, {3, 1}, {3, 0}, {2, 2}, {2, 1}, {2, 0}, {1, 5}, {1, 4}, {1, 3}, {1, 2}, {1, 1}, {1, 0}};
    for (auto &[major, minor] : versions)
    {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_RESIZABLE, is_resizable);
        glfwWindowHint(GLFW_VISIBLE, is_visible);
        glfwWindowHint(GLFW_DECORATED, !is_borderless);
        GLFWwindow *window = glfwCreateWindow(w, h, title, nullptr, (GLFWwindow *)share_ctx);
        glfwDefaultWindowHints();
        if (window)
        {
            if (x >= 0 && y >= 0)
            {
                glfwSetWindowPos(window, x, y);
            }
            return window;
        }
    }
    return nullptr;
}

auto hikari::platforms::glfw::WindowManager::createVulkanWindow(int w, int h, int x, int y, const char *title, bool is_visible, bool is_resizable, bool is_borderless) -> void *
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, is_resizable);
    glfwWindowHint(GLFW_VISIBLE, is_visible);
    glfwWindowHint(GLFW_DECORATED, !is_borderless);
    GLFWwindow *window = glfwCreateWindow(w, h, title, nullptr, nullptr);
    glfwDefaultWindowHints();
    if (window)
    {
        if (x >= 0 && y >= 0)
        {
            glfwSetWindowPos(window, x, y);
        }
    }
    return window;
}

auto hikari::platforms::glfw::WindowManager::createVulkanSurface(void *window, void *vk_instance) -> void *
{
    VkSurfaceKHR surface = nullptr;
    if (glfwCreateWindowSurface((VkInstance)vk_instance, (GLFWwindow *)window, nullptr, &surface) == VK_SUCCESS)
    {
        return surface;
    }
    return nullptr;
}

bool hikari::platforms::glfw::WindowManager::supportVulkan() const { return glfwVulkanSupported(); }

auto hikari::platforms::glfw::WindowManager::getVulkanSurfaceExtensionNames() const -> std::vector<const char *>
{
    auto required_extension_count = 0u;
    auto p_required_extension_names = glfwGetRequiredInstanceExtensions(&required_extension_count);
    auto required_extension_names = std::vector<const char *>(p_required_extension_names, p_required_extension_names + required_extension_count);
    return required_extension_names;
}

void hikari::platforms::glfw::WindowManager::destroyWindow(void *window)
{
    if (!window)
    {
        return;
    }
    glfwDestroyWindow((GLFWwindow *)window);
}

void hikari::platforms::glfw::WindowManager::setCurrentContext(void *ctx)
{
    if (ctx)
    {
        glfwMakeContextCurrent((GLFWwindow *)ctx);
    }
    else
    {
        glfwMakeContextCurrent((GLFWwindow *)nullptr);
    }
}

auto hikari::platforms::glfw::WindowManager::getCurrentContext() -> void *
{
    return glfwGetCurrentContext();
}

void hikari::platforms::glfw::WindowManager::pollEvents() { glfwPollEvents(); }
void hikari::platforms::glfw::WindowManager::swapBuffers(void* window)
{
  glfwSwapBuffers((GLFWwindow*)window);
}
bool hikari::platforms::glfw::WindowManager::shouldCloseWindow(void *window)
{
    if (!window)
    {
        return false;
    }
    return glfwWindowShouldClose((GLFWwindow *)window);
}

void hikari::platforms::glfw::WindowManager::showWindow(void *window)
{
    if (!window)
    {
        return;
    }
    glfwShowWindow((GLFWwindow *)window);
}

void hikari::platforms::glfw::WindowManager::hideWindow(void *window)
{
    if (!window)
    {
        return;
    }
    glfwHideWindow((GLFWwindow *)window);
}

hikari::platforms::glfw::WindowManager::WindowManager() noexcept { glfwInit(); }

auto hikari::platforms::glfw::WindowManager::getWindowUserPointer(void *window) -> void *
{
    return glfwGetWindowUserPointer((GLFWwindow *)window);
}
void hikari::platforms::glfw::WindowManager::setWindowUserPointer(void *window, void *p_data)
{
    return glfwSetWindowUserPointer((GLFWwindow *)window, p_data);
}

void hikari::platforms::glfw::WindowManager::setWindowSize(void *window, int w, int h)
{
    glfwSetWindowSize((GLFWwindow *)window, w, h);
}

void hikari::platforms::glfw::WindowManager::setWindowPosition(void *window, int x, int y)
{
    glfwSetWindowPos((GLFWwindow *)window, x, y);
}

void hikari::platforms::glfw::WindowManager::setWindowResizable(void *window, bool cond)
{
    glfwSetWindowAttrib((GLFWwindow *)window, GLFW_RESIZABLE, cond);
}

void hikari::platforms::glfw::WindowManager::getWindowSize(void *window, int &w, int &h)
{
    glfwGetFramebufferSize((GLFWwindow *)window, &w, &h);
}
void hikari::platforms::glfw::WindowManager::getWindowPosition(void *window, int &x, int &y)
{
    glfwGetFramebufferSize((GLFWwindow *)window, &x, &y);
}
void hikari::platforms::glfw::WindowManager::getFramebufferSize(void *window, int &w, int &h)
{
    glfwGetFramebufferSize((GLFWwindow *)window, &w, &h);
}

void hikari::platforms::glfw::WindowManager::setWindowVisible(void *window, bool cond)
{
    if (cond)
    {
        glfwShowWindow((GLFWwindow *)window);
    }
    else
    {
        glfwHideWindow((GLFWwindow *)window);
    }
}

void hikari::platforms::glfw::WindowManager::setWindowBorderless(void* window, bool borderless)
{
  glfwSetWindowAttrib((GLFWwindow*)window, GLFW_DECORATED, !borderless);
}

void hikari::platforms::glfw::WindowManager::setWindowIconified(void *window)
{
    glfwIconifyWindow((GLFWwindow *)window);
}

auto hikari::platforms::glfw::WindowManager::getClipboard(void *window) -> const char *
{
    return glfwGetClipboardString((GLFWwindow *)window);
}

void hikari::platforms::glfw::WindowManager::setClipboard(void *window, const char *name)
{
    glfwSetClipboardString((GLFWwindow *)window, name);
}
void hikari::platforms::glfw::WindowManager::setWindowTitle(void* window, const char* name)
{
  glfwSetWindowTitle((GLFWwindow*)window, name);
}


#define HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(NAME, ...) \
    void hikari::platforms::glfw::WindowManager::set##NAME(void *window, void (*callback)(GLFWwindow * handle, __VA_ARGS__))
#define HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK_VOID(NAME) \
    void hikari::platforms::glfw::WindowManager::set##NAME(void *window, void (*callback)(GLFWwindow * handle))

HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowSize, int32_t w, int32_t h) { glfwSetWindowSizeCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowPosition, int32_t x, int32_t y) { glfwSetWindowSizeCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK_VOID(CallbackWindowClose) { glfwSetWindowCloseCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackWindowIconified, int32_t iconified) { glfwSetWindowIconifyCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackFramebufferSize, int32_t w, int32_t h) { glfwSetFramebufferSizeCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackCursorPosition, double x, double y) { glfwSetCursorPosCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackKey, int32_t key, int32_t scancode, int32_t action, int32_t mods) { glfwSetKeyCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackChar, uint32_t codepoint) { glfwSetCharCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackMouseButton, int32_t button, int32_t action, int32_t mods) { glfwSetMouseButtonCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackCursorEnter, int32_t enter) { glfwSetCursorEnterCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackDrop, int path_count, const char *paths[]) { glfwSetDropCallback((GLFWwindow *)window, callback); }
HK_PLATFORM_GLFW_DEFINE_SET_NATIVE_CALLBACK(CallbackScroll, double x, double y) { glfwSetScrollCallback((GLFWwindow *)window, callback); }

using GetProcAddressType = hikari::platforms::glfw::gl_proc (*)(const char *name);
GetProcAddressType hikari::platforms::glfw::WindowManager::GetProcAddress = glfwGetProcAddress;
