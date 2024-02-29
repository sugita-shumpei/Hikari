#pragma once
#include <array>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <hikari/core/singleton.h>
#include <hikari/events/event.h>
#include <hikari/events/handler.h>
#include <hikari/events/target.h>
#include <hikari/thread/spin_lock.h>
namespace hikari {
  struct EventQueue;
  struct EventSystemImpl {
    bool initialize();
    void terminate ();
    void update(); //並列性の観点から, 非破壊操作は即時に実行されず, 次フレームにまとめて更新する
    bool isInitialized() const noexcept;
    auto getGlobalQueue() noexcept -> EventQueue&;// global event queue(最終的にEventはこのQueueに詰める )
    auto createTarget(const std::string& name) -> EventTarget;
    void destroyTarget(const EventTarget& target);
    void signal(const Event& event);// eventを処理する
    void dispatchOne(EventQueue& queue);// queueからEventを1つ取り出し, 処理する
    void dispatchAll(EventQueue& queue);// queueからEventを全て取り出し, 処理する
    void dispatchOne();// global queueからEventを1つ取り出し, 処理する(使い方要注意)
    void dispatchAll();// global queueからEventを全て取り出し, 処理する（） 
  private:
    friend class EventTarget;
    void addTargetHandler(const std::string& target_name, std::unique_ptr<EventHandler>&& handler);
    void removeTargetHandler(const std::string& target_name, EventType type);
    bool hasTargetHandler(const std::string& target_name, EventType type) const noexcept;
  private:
    bool m_is_initialized = false;
    tsl::robin_set<std::string> m_targets = {};
    std::array<tsl::robin_map<std::string, std::unique_ptr<EventHandler>>, (size_t)EventType::eCount> m_handlers = {};
  };
  using EventSystem = SingletonWithInitAndTerm<EventSystemImpl>;
}
