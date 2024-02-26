#pragma once
#include <cassert>
#include <atomic>
namespace hikari {
  struct SpinLock {
    SpinLock(bool locked = false) noexcept :m_is_locked{ locked } {}
    SpinLock(const SpinLock&) = delete;
    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    void lock() noexcept {
      // Acquire操作は, これより前の処理に行われる処理について順序保証を行う
      while (true) {
        const auto locked = m_is_locked.exchange(true, std::memory_order_acquire);
        if (!locked) { break; }
        wait();
      }
    }
    bool try_lock() noexcept {
      if (is_locked()) { return false; }
      const auto locked = m_is_locked.exchange(true, std::memory_order_acquire);
      return !locked;
    }
    void unlock() noexcept {
      assert(is_locked());
      m_is_locked.store(false, std::memory_order_release);
    }
    bool is_locked() const noexcept {
      return m_is_locked.load(std::memory_order_relaxed);
    }
  private:
    void wait();
  private:
    std::atomic<bool> m_is_locked ;
  };
}
