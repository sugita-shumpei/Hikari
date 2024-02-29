#pragma once
#include <hikari/events/event.h>
namespace hikari {

  HK_EVENT_BEG_DEFINE(AppRemoveWindow)
    AppRemoveWindowEvent(Window* window)noexcept : Event(), m_window{ window } {}
    auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eApplication; }
    auto getWindow() const-> Window* { return m_window; }
  private:
    Window* m_window;
  HK_EVENT_END_DEFINE(AppRemoveWindow);


  HK_EVENT_BEG_DEFINE(AppFinish)
    AppFinishEvent()noexcept : Event() {}
    auto getCategory() const->EventCategoryFlags override { return EventCategoryFlagBits::eApplication; }
  HK_EVENT_END_DEFINE(AppFinish);
}
