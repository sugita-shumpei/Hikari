#pragma once
#include <string>
#include <hikari/events/handler.h>
namespace hikari {
  struct EventTarget {
     EventTarget() noexcept {}
     EventTarget(const std::string& name) noexcept :m_name{ name } {}
     EventTarget(const EventTarget&) noexcept = delete;
     EventTarget& operator=(const EventTarget&) noexcept = delete;
     EventTarget(EventTarget&&) noexcept = default;
     EventTarget& operator=(EventTarget&&) noexcept = default;
    ~EventTarget() noexcept {}

    bool operator!() const noexcept { return !m_is_valid; }
    operator bool() const noexcept { return m_is_valid; }
    auto getName() const noexcept  -> const std::string& { return m_name; }
    auto getTypes() const noexcept -> const std::vector<EventType>& { return m_types; }
    void addHandler(std::unique_ptr<EventHandler>&& handler);
    void removeHandler(EventType type);
  private:
    friend class EventSystemImpl;
    std::string m_name = "";
    bool m_is_valid = false;
    std::vector<EventType> m_types = {};
  };
}
