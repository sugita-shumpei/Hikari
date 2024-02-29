#include <hikari/events/system.h>
#include <hikari/events/queue.h>

bool hikari::EventSystemImpl::initialize() {
  m_is_initialized = true;
  (void)getGlobalQueue();
  return true;
}

void hikari::EventSystemImpl::terminate() {
  m_is_initialized = false;
  auto& queue = getGlobalQueue();
  queue.clear();
  m_targets.clear();
  m_handlers = {};
}

void hikari::EventSystemImpl::update()
{

}

bool hikari::EventSystemImpl::isInitialized() const noexcept { return m_is_initialized; }

auto hikari::EventSystemImpl::getGlobalQueue() noexcept -> EventQueue&
{
  // TODO: return ステートメントをここに挿入します
  static EventQueue queue;
  return queue;
}

auto hikari::EventSystemImpl::createTarget(const std::string& name) -> EventTarget
{
  if (m_targets.count(name) != 0) { return EventTarget(); }
  // ここが破壊操作
  {
    m_targets.insert(name);
  }
  return EventTarget(name);
}

void hikari::EventSystemImpl::destroyTarget(const EventTarget& target)
{
  if (!target) { return; }
  auto& name = target.getName();
  auto& types = target.getTypes();
  for (auto& type : types) {
    removeTargetHandler(name, type);
  }
  // ここが破壊操作
  {
    m_targets.erase(name);
  }
}

void hikari::EventSystemImpl::signal(const Event& event) {
  auto type = event.getType();
  // 発火させるものの, 対象の優先順位などを考慮していない
  // 後で属性を追加する(優先順位の高いものから発火させる)
  for (auto& [name,handler] : m_handlers[(U32)type]) {
    handler->execute(event);
  }
}

void hikari::EventSystemImpl::dispatchOne(EventQueue& queue)
{
  auto&& event = queue.popOne();
  if (!event) { return; }
  if (event->isHandled()) { return; }
  signal(*event);
}

void hikari::EventSystemImpl::dispatchAll(EventQueue& queue)
{
  auto&& events = queue.popAll();
  if (events.empty()) { return; }
  for (auto& event : events) {// 重要なのは呼び出し順序ではなく, 優先順位順
    if (event->isHandled()) { continue; }
    signal(*event);
  }
}

void hikari::EventSystemImpl::dispatchOne()
{
  dispatchOne(getGlobalQueue());
}

void hikari::EventSystemImpl::dispatchAll()
{
  dispatchAll(getGlobalQueue());
}

void hikari::EventSystemImpl::addTargetHandler(const std::string& target_name, std::unique_ptr<EventHandler>&& handler)
{
  if (!handler) { return; }
  auto event_type = handler->getType();
  auto& handlers = m_handlers[(U32)event_type];
  {
    m_handlers[(U32)event_type].insert_or_assign(target_name, std::move(handler));// handlerを削除する
  }
}

void hikari::EventSystemImpl::removeTargetHandler(const std::string& target_name, EventType type)
{
  auto& handlers = m_handlers[(U32)type];
  auto iter = handlers.find(target_name);
  if (iter != handlers.end()) 
  {
    m_handlers[(U32)type].erase(iter);
  }
}

bool hikari::EventSystemImpl::hasTargetHandler(const std::string& target_name, EventType type) const noexcept
{
  return m_handlers[(U32)type].count(target_name) != 0;
}
