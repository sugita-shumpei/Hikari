#pragma once
#include <hikari/core/data_types.h>
#include <hikari/core/flags.h>

#define HK_EVENT_BEG_DEFINE(TYPE) \
  struct TYPE##Event : public Event { \
    static inline constexpr EventType kType = EventType::e##TYPE; \
    virtual ~TYPE##Event() noexcept{} \
    auto getType() const -> EventType override { return kType; }

#define HK_EVENT_END_DEFINE(TYPE) }

namespace hikari {
  enum class EventType : U8 {
    eAppTick,            // 一定時間ごとに呼び出される
    eAppUpdate,          // 毎フレームの更新時に呼び出される
    eAppUpdateLate,      // Updateの後最後に呼びされる
    eAppStart,           // アプリケーションの開始時に呼び出される
    eAppRender,          // アプリケーションの描画時に呼び出される(正確には描画コマンドそのものを呼び出すことはできないため, 次以降の描画コマンドを作成するのに使う)
    eAppFinish,          // アプリケーションの終了時に呼び出される
    eAppRemoveWindow,    // アプリケーションが画面を閉じるときに呼び出される
    eWindowClose,        // Windowが閉じるときに呼び出される
    eWindowDestroy,      // Windowが削除されるときに呼び出される 
    eWindowResize,       // Windowがリサイズされるときに呼び出される
    eWindowFullscreen,   // Windowがフルスクリーンになるとき呼び出される
    eWindowFocus,        // WindowにFocusしているとき呼び出される
    eWindowLostFocus,    // WindowのFocusが外れたとき呼び出される
    eWindowHide,         // Windowが隠れたとき呼び出される
    eWindowShow,         // Windowが表示されたとき呼び出される
    eWindowMoved,        // Windowが移動したとき呼び出される
    eKeyPressed,         // キーが押されているとき呼び出される
    eKeyReleased,        // キーが離れたとき呼び出される
    eKeyRepeated,        // キーが繰り返し押されているとき呼び出される
    eMouseButtonPressed, // マウスのボタンが押されているとき呼び出される
    eMouseButtonReleased,// マウスのボタンが離れているとき呼び出される
    eMouseMoved,         // マウスが動いたとき呼び出される
    eMouseScroll,        // マウスがスクロールしているとき呼び出される
    eCount
  };

  enum class EventCategoryFlagBits : U8 {
    eNone        = 0,
    eApplication = 1,
    eWindow      = 2,
    eInput       = 4,
    eKeyboard    = 8,
    eMouse       = 16,
    eMouseButton = 32,
    eMask        = 63
  };

  template<>
  struct FlagsTraits<EventCategoryFlagBits> :
  FlagsTraitsDefineUtils<EventCategoryFlagBits>{};
  using EventCategoryFlags = Flags<EventCategoryFlagBits>;

  struct Event {
  public:
    virtual auto getType()     const -> EventType          = 0;
    virtual auto getCategory() const -> EventCategoryFlags = 0;
    virtual ~Event() noexcept {}
    bool isHandled() const noexcept { return handled; }
    void setHandled() noexcept { handled = true; }
  private:
    bool handled = false;
  };

}
