#pragma once
#include <queue>
#include <hikari/thread/spin_lock.h>
#include <hikari/events/event.h>
#include <memory>
#include <mutex>
namespace hikari {
  // Eventを即時実行する代わりにQueueに詰めて, 実行することが可能
  // あくまでデータ構造であって, 操作は行わない点に注意
  // 注意: Lock中にロックすると死ぬので, EventHandler内でEventQueueを触ることは禁止
  // なんでも入るQueueなのが非常に邪悪
  struct EventQueue{
    EventQueue() noexcept {}
    void push(std::unique_ptr<Event>&& event)
    {
      std::lock_guard lk(m_lock);
      m_events.push(std::move(event));
    }
    auto popOne() -> std::unique_ptr<Event>
    {
      std::lock_guard lk(m_lock);
      auto event = std::move(m_events.front());
      m_events.pop();
      return event;
    }
    auto popAll() -> std::vector<std::unique_ptr<Event>> 
    {
      std::lock_guard lk(m_lock);
      std::vector<std::unique_ptr<Event>> events = {};
      events.reserve(m_events.size());
      while(m_events.size() > 0){
        events.push_back(std::move(m_events.front()));
        m_events.pop();
      }
      return events;
    }
    void clear()  {
      std::lock_guard lk(m_lock);
      m_events = {};
    }
    U64  size() const  {
      std::lock_guard lk(m_lock);
      return m_events.size();
    }
  private:
    mutable SpinLock m_lock = {};
    std::queue<std::unique_ptr<Event>> m_events = {};
  };
}
