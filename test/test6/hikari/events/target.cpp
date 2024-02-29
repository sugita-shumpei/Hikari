#include <ranges>
#include <algorithm>
#include <hikari/events/target.h>
#include <hikari/events/system.h>
void hikari::EventTarget::addHandler(std::unique_ptr<EventHandler>&& handler)
{
  EventSystem::getInstance()->addTargetHandler(m_name, std::move(handler));
}
void hikari::EventTarget::removeHandler(EventType type)
{
  EventSystem::getInstance()->removeTargetHandler(m_name, type);
  auto iter = std::ranges::find(m_types, type);
  if (iter != std::end(m_types)) {
    m_types.erase(iter);
  }
}
