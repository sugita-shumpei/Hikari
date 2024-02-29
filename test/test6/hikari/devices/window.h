#pragma once
#include <hikari/core/singleton.h>
#include <hikari/core/data_types.h>
#include <hikari/core/platform.h>
#include <hikari/devices/common.h>
#include <hikari/events/target.h>
namespace hikari {
  struct EventTarget;
  struct Monitor;
  struct Window {
    using FlagBits = WindowFlagBits;
    using Flags = WindowFlags;
    auto getNativeWindow() const -> void*;
    auto getSize() const-> UVec2;
    auto getSurfaceSize() const->UVec2;
    auto getPosition() const->IVec2;
    void update();// この際, Event Managerに対してEvent送信を行う
    bool willDestroy() const noexcept;// 
  private:
    friend class DeviceSystemImpl;
     Window(const WindowCreateDesc& desc) noexcept;
    ~Window() noexcept;
    void onDestroy();
  private:
    GLFWwindow* m_handle;
    struct Data {
      Window*     window;
      UVec2       size;
      UVec2       surface_size;
      IVec2       position;
      std::string title;
      Flags       flags;
    } m_data;
    EventTarget m_target;
    bool  m_will_destroy;
  };
}
