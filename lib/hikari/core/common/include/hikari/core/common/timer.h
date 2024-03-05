#pragma once
#include <chrono>
namespace hikari {
  namespace core {
    struct Timer {
      Timer(double offset = 0.0f) noexcept : m_start{ decltype(m_start)::clock::now() }, m_offset{ offset } {}
      ~Timer() noexcept {}
      double getTime() const {
        auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          decltype(m_start)::clock::now() - m_start
        ).count();
        return m_offset + time_ns * 0.001 * 0.001 * 0.001;
      }
      double setTime(double new_offset = 0.0)  {
        auto prv_start = m_start;
        auto prv_offset = m_offset;
        m_start  = decltype(m_start)::clock::now();
        m_offset = new_offset;

        auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          m_start - prv_start
        ).count();
        return prv_offset + time_ns * 0.001 * 0.001 * 0.001;
      }
    private:
      std::chrono::high_resolution_clock::time_point m_start;
      double m_offset;
    };
  }
}
