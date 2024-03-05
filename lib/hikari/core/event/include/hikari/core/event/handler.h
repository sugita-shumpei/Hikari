#pragma once
#include <function2/function2.hpp>
#include <hikari/core/common/unique_id.h>
#include <hikari/core/event/event.h>
namespace hikari {
  namespace core {
    struct IEvent;
    struct IEventHandler {
      IEventHandler() noexcept {}
      virtual ~IEventHandler() noexcept {}
      void execute(const IEvent& event) { call(event); }
      virtual auto getID() const noexcept -> UniqueID = 0;
      virtual auto getType() const noexcept -> EventType = 0;
    private:
      virtual void call(const IEvent& event) = 0;
    };

    template<typename EventT>
    using  EventHandlerCallback = fu2::function<void(const EventT&)>;

    template<typename EventT>
    using  EventHandlerCallbackView = fu2::function_view<void(const EventT&)>;

    template<typename EventT>
    struct EventHandlerWithCallback : public IEventHandler {
      using Type = EventType;
      static inline constexpr Type kType = EventT::kType;
      EventHandlerWithCallback(EventHandlerCallback<EventT>&& func) noexcept
        :m_callback{ std::move(func) } , m_ID(){}
      virtual ~EventHandlerWithCallback() noexcept {}
      auto getID() const noexcept -> UniqueID override { return m_ID.getID(); }
      auto getType() const noexcept -> EventType override { return EventT::kType; }
    private:
      void call(const IEvent& event) override {
        if (event.getType() == EventT::kType) { m_callback(static_cast<const EventT&>(event)); }
      }
    private:
      EventHandlerCallback<EventT> m_callback;
      UniqueIDGenerator m_ID;
    };

    template<typename EventT>
    struct EventHandlerWithCallbackView : public IEventHandler {
      using Type = EventType;
      static inline constexpr Type kType = EventT::kType;
      EventHandlerWithCallbackView(EventHandlerCallbackView<EventT>&& func) noexcept
        :m_callback{ std::move(func) }, m_ID() {}
      virtual ~EventHandlerWithCallbackView() noexcept {}
      auto getID() const noexcept -> UniqueID override { return m_ID.getID(); }
      auto getType() const noexcept -> EventType override { return EventT::kType; }
    private:
      void call(const IEvent& event) override {
        if (event.getType() == EventT::kType) { m_callback(static_cast<const EventT&>(event)); }
      }
    private:
      EventHandlerCallbackView<EventT> m_callback;
      UniqueIDGenerator m_ID;
    };

    template<typename EventT>
    inline auto makeUniqueEventHandlerFromCallback(EventHandlerCallback<EventT>&& func) -> std::unique_ptr<IEventHandler> {
      return std::unique_ptr<IEventHandler>(new EventHandlerWithCallback<EventT>(std::move(func)));
    }
    template<typename EventT>
    inline auto makeUniqueEventHandlerFromCallbackView(EventHandlerCallbackView<EventT>&& func) -> std::unique_ptr<IEventHandler> {
      return std::unique_ptr<IEventHandler>(new EventHandlerWithCallbackView<EventT>(std::move(func)));
    }
  }
}
