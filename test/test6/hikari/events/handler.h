#pragma once
#include <function2/function2.hpp>
#include <hikari/events/event.h>
namespace hikari {
  // Event Handler(特定のEventに対してHandleする)
  struct EventHandler {
    virtual ~EventHandler() noexcept {}
    inline  void execute(const Event& e) { call(e); }
    virtual auto getType() const->EventType  = 0;
  protected:
    virtual void call(const Event& e) = 0;
  };

  template<typename EventT>
  using EventHandlerCallback = std::function<void(const EventT&)>;

  template<typename EventT>
  struct TypeEventHandler : public EventHandler {
    explicit TypeEventHandler(const EventHandlerCallback<EventT>& callback)
      :m_callback{callback}{}
    virtual ~TypeEventHandler() noexcept {}
    virtual auto getType() const -> EventType { return EventT::kType; }
  private:
    void call(const Event& e) override {
      if (e.getType() == EventT::kType) { return m_callback(static_cast<const EventT&>(e)); }
    }
    EventHandlerCallback<EventT> m_callback;
  };

  template<typename EventT>
  inline auto makeUniqueEventHandler(const EventHandlerCallback<EventT>& callback) -> std::unique_ptr<EventHandler> {
    return std::unique_ptr< EventHandler>(new TypeEventHandler(callback));
  }

}
