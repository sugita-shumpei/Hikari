#pragma once
#include <hikari/core/event/event.h>
#include <hikari/core/input/common.h>
namespace hikari {
  namespace core {
    HK_EVENT_BEG_DEFINE(KeyPressed);
    KeyInput key = {};
    KeyModFlags mod = {};
    HK_EVENT_END_DEFINE(KeyPressed);

    HK_EVENT_BEG_DEFINE(KeyReleased);
    KeyInput key = {};
    KeyModFlags mod = {};
    HK_EVENT_END_DEFINE(KeyReleased);

    HK_EVENT_BEG_DEFINE(KeyRepeated);
    KeyInput key = {};
    KeyModFlags mod = {};
    HK_EVENT_END_DEFINE(KeyRepeated);

    HK_EVENT_BEG_DEFINE(MouseButtonPressed);
    MouseButtonInput button = {};
    KeyModFlags mod = {};
    HK_EVENT_END_DEFINE(MouseButtonPressed);

    HK_EVENT_BEG_DEFINE(MouseButtonReleased);
    MouseButtonInput button = {};
    KeyModFlags mod = {};
    HK_EVENT_END_DEFINE(MouseButtonReleased);

    HK_EVENT_BEG_DEFINE(MouseMoved);
    Offset2<F64> position = {};
    HK_EVENT_END_DEFINE(MouseMoved);

    HK_EVENT_BEG_DEFINE(MouseScrolled);
    Offset2<F64> offset = {};
    HK_EVENT_END_DEFINE(MouseScrolled);
  }
}
