#pragma once
#include <variant>
#include <hikari/core/window/common.h>
#include <hikari/core/event/event.h>
namespace hikari {
  namespace core {
    HK_EVENT_BEG_DEFINE(WindowClose);
    Window* window = nullptr;
    HK_EVENT_END_DEFINE(WindowClose);

    HK_EVENT_BEG_DEFINE(WindowResize);
    Window* window = nullptr;
    WindowExtent2D size = {};
    HK_EVENT_END_DEFINE(WindowResize);

    HK_EVENT_BEG_DEFINE(WindowMoved);
    Window* window = nullptr;
    WindowOffset2D position = {};
    HK_EVENT_END_DEFINE(WindowMoved);

    HK_EVENT_BEG_DEFINE(WindowFocus);
    Window* window = nullptr;
    HK_EVENT_END_DEFINE(WindowFocus);

    HK_EVENT_BEG_DEFINE(WindowLeave);
    Window* window = nullptr;
    HK_EVENT_END_DEFINE(WindowLeave);

    HK_EVENT_BEG_DEFINE(WindowIconified);
    Window* window = nullptr;
    HK_EVENT_END_DEFINE(WindowIconified);

    HK_EVENT_BEG_DEFINE(WindowRestored);
    Window* window = nullptr;
    HK_EVENT_END_DEFINE(WindowRestored);

    HK_EVENT_BEG_DEFINE(WindowFullscreen);
    Window* window = nullptr;
    WindowExtent2D size = {};
    HK_EVENT_END_DEFINE(WindowFullscreen);

    HK_EVENT_BEG_DEFINE(WindowWindowed);
    Window* window = nullptr;
    WindowExtent2D size = {};
    WindowOffset2D position = {};
    HK_EVENT_END_DEFINE(WindowWindowed);
  }
}
