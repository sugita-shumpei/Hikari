#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace hikari {
  namespace core {
    struct GLFWContext {
      static auto getInstance() noexcept -> GLFWContext& {
        static GLFWContext ctx; return ctx;
      }
      ~GLFWContext() noexcept {}
      void addRef() noexcept {
        if (m_counter == 0) {
          glfwInit();
        }
        ++m_counter;
      }
      void release() {
        if (m_counter == 0) {
          return;
        }
        if (m_counter == 1) {
          glfwTerminate();
        }
        --m_counter;
      }
    private:
      GLFWContext() noexcept {}
      GLFWContext(const GLFWContext&) noexcept = delete;
      GLFWContext& operator=(const GLFWContext&) noexcept = delete;
      GLFWContext(GLFWContext&&) noexcept = delete;
      GLFWContext& operator=(GLFWContext&&) noexcept = delete;
      // main threadで呼ばないといけないため, 単純なカウンターでよい
    private:
      uint32_t m_counter = 0;
    };

  }
}
