#pragma once
#include <hikari/graphics/instance.h>
#include <hikari/graphics/opengl/common.h>
namespace hikari {
  struct GraphicsSystem;
  struct GraphicsOpenGLInstance : public GraphicsInstance {
    static inline constexpr GraphicsAPIType kAPIType = GraphicsAPIType::eOpenGL;
    virtual ~GraphicsOpenGLInstance() noexcept;
    auto getHandle() const -> GLFWwindow*;
    virtual auto getAPIType() const->GraphicsAPIType override { return GraphicsAPIType::eOpenGL; }
    bool isInitialized() const override;
  private:
    friend struct GraphicsSystem;
    GraphicsOpenGLInstance() noexcept;
    bool initialize() override;
    void terminate() override;
  private:
    GLFWwindow* m_window  = nullptr;
    bool m_is_initialized = false;
  };
}
