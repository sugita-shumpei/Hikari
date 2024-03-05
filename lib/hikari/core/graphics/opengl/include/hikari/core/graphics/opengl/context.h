#pragma once
#include <thread>
#include <glad/glad.h>
namespace hikari {
  namespace core {
    struct Window;
    struct GraphicsOpenGLRenderer;
    struct GraphicsOpenGLContext {
      ~GraphicsOpenGLContext() noexcept;
      // 同期スレッドから使用スレッドに初期化命令を投げる
      bool isOwnerThread(const  std::thread::id& thid) noexcept { return m_thid == thid; }
      void registerThread();
      auto getThreadID() const noexcept -> std::thread::id;
      void setCurrent();
      void popCurrent();
      auto getHandle()const -> void*;
    private:
      friend class GraphicsOpenGLRenderer;
      GraphicsOpenGLContext(GraphicsOpenGLRenderer* renderer, void* handle);
    private:
      GraphicsOpenGLRenderer* m_renderer = nullptr;
      void* m_handle = nullptr;
      std::thread::id m_thid{};
    };
  }
}
