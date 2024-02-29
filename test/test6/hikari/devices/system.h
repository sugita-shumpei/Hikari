#pragma once
#include <vector>
#include <hikari/core/singleton.h>
#include <hikari/core/data_types.h>
#include <hikari/devices/common.h>
namespace hikari {
  struct Monitor;
  struct Window;
  struct DeviceSystemImpl {
     DeviceSystemImpl() noexcept;
    ~DeviceSystemImpl() noexcept;
    bool isInitialized() const noexcept;
    bool initialize() ;
    void terminate() noexcept;
    // Window: 直ちに新しいWindowを作成する
    auto createWindow (const WindowCreateDesc& desc = {}) -> Window*;
    void destroyWindow(Window* window);
    // Monitor
    auto getMonitor(U32 idx) -> Monitor* { return nullptr; }
    auto getMonitorCount() -> U32 { return 0; }
    // Update: 全ての依存処理はこのタイミングでアップデートが実行される
    void update();
  private:
    bool m_is_initialized = false;
    std::vector<Window*> m_windows = {};
  };
  // Deviceシステムは, シングルトンでよい
  using DeviceSystem = SingletonWithInitAndTerm<DeviceSystemImpl>;
}
