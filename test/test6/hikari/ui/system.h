#pragma once
#include <GLFW/glfw3.h>
#include <hikari/graphics/common.h>
namespace hikari {
  struct UISystem {
    static auto getInstance() -> UISystem&;
    ~UISystem() noexcept {}
    bool isInitialized();
    auto getGraphicsAPIType() const -> GraphicsAPIType;
  private:
  private:
    UISystem() noexcept {}
    UISystem(const UISystem&) noexcept = delete;
    UISystem(UISystem&&) noexcept = delete;
    UISystem& operator=(const UISystem&) noexcept = delete;
    UISystem& operator=(UISystem&&) noexcept = delete;
  private:
    bool m_is_initialized = false;
    GLFWwindow* m_main_window = nullptr;
    GraphicsAPIType m_graphics_api = GraphicsAPIType::eUnknown;
  };
}
