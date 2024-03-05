#pragma once
#include <hikari/core/common/data_type.h>
#include <cstdint>
#include <concepts>
#define HK_EVENT_BEG_DEFINE(TYPE) \
struct TYPE##Event : public IEvent { \
  static inline constexpr EventType kType =  EventType::e##TYPE; \
  virtual ~TYPE##Event() noexcept {} \
  auto getType() const noexcept -> EventType { return kType; } \

#define HK_EVENT_END_DEFINE(TYPE) }

namespace hikari {
  namespace core {
    enum class EventType : U8 {
      eNone        ,
      // App
      eAppUpdate,
      eAppTickUpdate,
      eAppLateUpdate,
      // Window
      eWindowClose ,
      eWindowResize,
      eWindowFocus ,
      eWindowLeave ,
      eWindowMoved ,
      eWindowIconified,
      eWindowRestored,
      eWindowFullscreen,
      eWindowWindowed,
      // Renderer
      eRendererFinishFrame,// RendererのFrame処理が終了(ただし描画が完了しているとは限らない点に注意)
      eRendererQuit,// Rendererの処理終了時に呼ばれる
      // Key
      eKeyPressed ,
      eKeyReleased,
      eKeyRepeated,
      // Mouse
      eMouseButtonPressed,
      eMouseButtonReleased,
      eMouseMoved,
      eMouseScrolled
    };
    struct IEvent {
      using Type = EventType;
      virtual ~IEvent() noexcept {}
      virtual auto getType() const noexcept -> Type = 0;
    };

    HK_EVENT_BEG_DEFINE(None);
    HK_EVENT_END_DEFINE(None);
  }
}
