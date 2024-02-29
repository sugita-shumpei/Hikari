#pragma once
#include <hikari/events/event.h>
namespace hikari {
  struct Window;
  HK_EVENT_BEG_DEFINE(WindowResize)
    WindowResizeEvent(Window* window, const UVec2& size)noexcept : Event(), m_window{ window }, m_size{ size } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
  auto getSize() const -> UVec2 { return m_size; }
private:
  UVec2 m_size;
  Window* m_window;
  HK_EVENT_END_DEFINE(windowResize);

  HK_EVENT_BEG_DEFINE(WindowMoved)
    WindowMovedEvent(Window* window, const IVec2& pos)noexcept : Event(), m_window{ window }, m_position{ pos } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
  auto getPosition() const -> IVec2 { return m_position; }
private:
  IVec2 m_position;
  Window* m_window;
  HK_EVENT_END_DEFINE(WindowMoved);

  HK_EVENT_BEG_DEFINE(WindowClose)
    WindowCloseEvent(Window* window)noexcept : Event(), m_window{ window } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
private:
  Window* m_window;
  HK_EVENT_END_DEFINE(WindowClose);

  HK_EVENT_BEG_DEFINE(WindowShow)
    WindowShowEvent(Window* window)noexcept : Event(), m_window{ window } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
private:
  Window* m_window;
  HK_EVENT_END_DEFINE(WindowShow);

  HK_EVENT_BEG_DEFINE(WindowDestroy)
    WindowDestroyEvent(Window* window)noexcept : Event(), m_window{ window } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
private:
  Window* m_window;
  HK_EVENT_END_DEFINE(WindowDestroy);


  HK_EVENT_BEG_DEFINE(WindowHide)
    WindowHideEvent(Window* window)noexcept : Event(), m_window{ window } {}
  auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eWindow; }
  auto getWindow() const-> Window* { return m_window; }
private:
  Window* m_window;
  HK_EVENT_END_DEFINE(WindowHide);
}
