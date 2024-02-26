#pragma once
#include <vector>
namespace hikari {
  struct DeviceSystem {
    static auto getInstance() noexcept -> DeviceSystem&;
    ~DeviceSystem() noexcept;
  private:
    DeviceSystem() noexcept;
    DeviceSystem(const DeviceSystem&) noexcept = delete;
    DeviceSystem(DeviceSystem&&) noexcept = delete;
    DeviceSystem& operator=(const DeviceSystem&) noexcept = delete;
    DeviceSystem& operator=(DeviceSystem&&) noexcept = delete;
  };
}
