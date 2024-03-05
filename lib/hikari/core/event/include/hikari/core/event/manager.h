#pragma once
#include <list>
#include <memory>
#include <queue>
#include <tuple>
#include <concepts>
#include <hikari/core/event/event.h>
#include <hikari/core/event/handler.h>
namespace hikari {
  namespace core {
    template <std::derived_from<IEvent> EventT>
    struct SingleEventManager {
      SingleEventManager() noexcept {}
      SingleEventManager(SingleEventManager&&) noexcept = delete;
      SingleEventManager(const SingleEventManager&) noexcept = delete;
      SingleEventManager& operator=(SingleEventManager&&) noexcept = delete;
      SingleEventManager& operator=(const SingleEventManager&) noexcept = delete;
      ~SingleEventManager() noexcept {}
      UniqueID subscribe(std::unique_ptr<IEventHandler>&& handler)
      {
        if (!handler) { return 0; }
        if (handler->getType() != EventT::kType) { return 0; }
        auto uid = handler->getID();
        m_handlers.push_back(std::move(handler));
        return uid; 
      }
      void unsubscribe(UniqueID uid) {
        auto iter = std::ranges::find_if(
          m_handlers, [uid](const auto& p) {
            return p->getID() == uid;
          }
        );
        if (iter != std::end(m_handlers)) {
          m_handlers.erase(iter);
        }
      }
      void triggerEvent(const EventT& event) {
        for (auto& handler : m_handlers) {
          handler->execute(event);
        }
      }
      void submitEvent(std::unique_ptr<EventT>&& event) {
        m_queue.emplace(std::move(event));
      }
      void dispatchEvents()
      {
        while (!m_queue.empty()) {
          auto event = std::move(m_queue.front());
          m_queue.pop();
          triggerEvent(*event);
        }
      }
    private:
      std::list<std::unique_ptr<IEventHandler>> m_handlers = {};
      std::queue<std::unique_ptr<EventT>> m_queue = {};
    };
    struct EventManager {
      EventManager() noexcept {}
      EventManager(EventManager&&) noexcept = delete;
      EventManager(const EventManager&) noexcept = delete;
      EventManager& operator=(EventManager&&) noexcept = delete;
      EventManager& operator=(const EventManager&) noexcept = delete;
      ~EventManager() noexcept {}

      UniqueID subscribe(std::unique_ptr<IEventHandler>&& handler)
      {
        if (!handler) { return 0; }
        auto uid = handler->getID();
        m_handlers.push_back(std::move(handler));
        return uid;
      }
      void unsubscribe(UniqueID uid) {
        auto iter = std::ranges::find_if(
          m_handlers, [uid](const auto& p) {
            return p->getID() == uid;
          }
        );
        if (iter != std::end(m_handlers)) {
          m_handlers.erase(iter);
        }
      }
      void triggerEvent(const IEvent& event) {
        for (auto& handler : m_handlers) {
          handler->execute(event);
        }
      }
      void submitEvent(std::unique_ptr<IEvent>&& event) {
        m_queue.emplace(std::move(event));
      }
      void dispatchEvents()
      {
        while (!m_queue.empty()) {
          auto event = std::move(m_queue.front());
          m_queue.pop();
          triggerEvent(*event);
        }
      }
    private:
      std::list<std::unique_ptr<IEventHandler>> m_handlers = {};
      std::queue<std::unique_ptr<IEvent>> m_queue = {};
    };
  }
}
