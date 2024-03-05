#pragma once
#include <vector>
#include <hikari/core/common/data_type.h>
#include <hikari/core/event/handler.h>
#include <hikari/core/event/manager.h>
namespace hikari {
  namespace core {
    struct EventTarget {
       EventTarget(EventManager& manager) noexcept : m_manager{ manager }, m_ids{} {}
      ~EventTarget() noexcept {
        for (auto& uid : m_ids) {
          m_manager.unsubscribe(uid);
        }
        m_ids.clear();
      }

      UniqueID subscribe(std::unique_ptr<IEventHandler>&& handler) {
        auto uid = m_manager.subscribe(std::move(handler));
        m_ids.push_back(uid);
        return uid;
      }
      void unsubscribe(UniqueID uid) {
        if (std::ranges::find(m_ids, uid) != std::end(m_ids)) {
          m_manager.unsubscribe(uid);
        }
      }
    private:
      EventManager& m_manager;
      std::vector<UniqueID> m_ids;
    };
  }
}
