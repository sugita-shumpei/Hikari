#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
namespace hikari {
  namespace core {
    using  UniqueID = std::uint32_t;
    struct UniqueIDGenerator {
       UniqueIDGenerator() noexcept :m_ID{0u}{}
       UniqueIDGenerator(const UniqueIDGenerator& lhs) noexcept = delete;
       UniqueIDGenerator& operator=(UniqueIDGenerator& lhs) noexcept = delete;
       UniqueIDGenerator(UniqueIDGenerator&& rhs) noexcept :m_ID{ std::exchange(m_ID,0u) } {}
       UniqueIDGenerator& operator=(UniqueIDGenerator&& rhs) noexcept { if (this != &rhs) { m_ID = std::exchange(m_ID, 0u); } return *this; }
      ~UniqueIDGenerator() noexcept {}
      operator UniqueID () const noexcept { return getID(); }
      auto getID() const noexcept -> UniqueID {
        if (m_ID == 0u) {
          static std::atomic<UniqueID> id_generator{ 0u };
          m_ID = id_generator.fetch_add(+1u) + 1u;
          return m_ID;
        }
        return m_ID;
      }
    private:
      mutable UniqueID m_ID;
    };
  }
}
