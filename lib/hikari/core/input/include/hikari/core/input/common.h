#pragma once
#include <hikari/core/common/data_type.h>
#include <hikari/core/common/flags.h>

#define HK_DECLARE_KEY_INPUT(KEY, GLFW_KEY) KEY

namespace hikari {
  namespace core {
    enum class KeyInput :U8
    {
      HK_DECLARE_KEY_INPUT(eSpace, SPACE),
      HK_DECLARE_KEY_INPUT(eApostrophe, APOSTROPHE),
      HK_DECLARE_KEY_INPUT(eComma, COMMA),
      HK_DECLARE_KEY_INPUT(eMinus, MINUS),
      HK_DECLARE_KEY_INPUT(ePeriod, PERIOD),
      HK_DECLARE_KEY_INPUT(eSlash, SLASH),
      HK_DECLARE_KEY_INPUT(e0, 0),
      HK_DECLARE_KEY_INPUT(e1, 1),
      HK_DECLARE_KEY_INPUT(e2, 2),
      HK_DECLARE_KEY_INPUT(e3, 3),
      HK_DECLARE_KEY_INPUT(e4, 4),
      HK_DECLARE_KEY_INPUT(e5, 5),
      HK_DECLARE_KEY_INPUT(e6, 6),
      HK_DECLARE_KEY_INPUT(e7, 7),
      HK_DECLARE_KEY_INPUT(e8, 8),
      HK_DECLARE_KEY_INPUT(e9, 9),
      HK_DECLARE_KEY_INPUT(eSemicolon, SEMICOLON),
      HK_DECLARE_KEY_INPUT(eEqual, EQUAL),
      HK_DECLARE_KEY_INPUT(eA, A),
      HK_DECLARE_KEY_INPUT(eB, B),
      HK_DECLARE_KEY_INPUT(eC, C),
      HK_DECLARE_KEY_INPUT(eD, D),
      HK_DECLARE_KEY_INPUT(eE, E),
      HK_DECLARE_KEY_INPUT(eF, F),
      HK_DECLARE_KEY_INPUT(eG, G),
      HK_DECLARE_KEY_INPUT(eH, H),
      HK_DECLARE_KEY_INPUT(eI, I),
      HK_DECLARE_KEY_INPUT(eJ, J),
      HK_DECLARE_KEY_INPUT(eK, K),
      HK_DECLARE_KEY_INPUT(eL, L),
      HK_DECLARE_KEY_INPUT(eM, M),
      HK_DECLARE_KEY_INPUT(eN, N),
      HK_DECLARE_KEY_INPUT(eO, O),
      HK_DECLARE_KEY_INPUT(eP, P),
      HK_DECLARE_KEY_INPUT(eQ, Q),
      HK_DECLARE_KEY_INPUT(eR, R),
      HK_DECLARE_KEY_INPUT(eS, S),
      HK_DECLARE_KEY_INPUT(eT, T),
      HK_DECLARE_KEY_INPUT(eU, U),
      HK_DECLARE_KEY_INPUT(eV, V),
      HK_DECLARE_KEY_INPUT(eW, W),
      HK_DECLARE_KEY_INPUT(eX, X),
      HK_DECLARE_KEY_INPUT(eY, Y),
      HK_DECLARE_KEY_INPUT(eZ, Z),
      HK_DECLARE_KEY_INPUT(eLeft_Bracket, LEFT_BRACKET),
      HK_DECLARE_KEY_INPUT(eBackslash, BACKSLASH),
      HK_DECLARE_KEY_INPUT(eRight_Bracket, RIGHT_BRACKET),
      HK_DECLARE_KEY_INPUT(eGrave_Accent, GRAVE_ACCENT),
      HK_DECLARE_KEY_INPUT(eWorld_1, WORLD_1),
      HK_DECLARE_KEY_INPUT(eWorld_2, WORLD_2),
      HK_DECLARE_KEY_INPUT(eEscape, ESCAPE),
      HK_DECLARE_KEY_INPUT(eEnter, ENTER),
      HK_DECLARE_KEY_INPUT(eTab, TAB),
      HK_DECLARE_KEY_INPUT(eBackspace, BACKSPACE),
      HK_DECLARE_KEY_INPUT(eInsert, INSERT),
      HK_DECLARE_KEY_INPUT(eDelete, DELETE),
      HK_DECLARE_KEY_INPUT(eRight, RIGHT),
      HK_DECLARE_KEY_INPUT(eLeft, LEFT),
      HK_DECLARE_KEY_INPUT(eDown, DOWN),
      HK_DECLARE_KEY_INPUT(eUp, UP),
      HK_DECLARE_KEY_INPUT(ePage_Up, PAGE_UP),
      HK_DECLARE_KEY_INPUT(ePage_Down, PAGE_DOWN),
      HK_DECLARE_KEY_INPUT(eHome, HOME),
      HK_DECLARE_KEY_INPUT(eEnd, END),
      HK_DECLARE_KEY_INPUT(eCaps_Lock, CAPS_LOCK),
      HK_DECLARE_KEY_INPUT(eScroll_Lock, SCROLL_LOCK),
      HK_DECLARE_KEY_INPUT(eNum_Lock, NUM_LOCK),
      HK_DECLARE_KEY_INPUT(ePrint_Screen, PRINT_SCREEN),
      HK_DECLARE_KEY_INPUT(ePause, PAUSE),
      HK_DECLARE_KEY_INPUT(eF1, F1),
      HK_DECLARE_KEY_INPUT(eF2, F2),
      HK_DECLARE_KEY_INPUT(eF3, F3),
      HK_DECLARE_KEY_INPUT(eF4, F4),
      HK_DECLARE_KEY_INPUT(eF5, F5),
      HK_DECLARE_KEY_INPUT(eF6, F6),
      HK_DECLARE_KEY_INPUT(eF7, F7),
      HK_DECLARE_KEY_INPUT(eF8, F8),
      HK_DECLARE_KEY_INPUT(eF9, F9),
      HK_DECLARE_KEY_INPUT(eF10, F10),
      HK_DECLARE_KEY_INPUT(eF11, F11),
      HK_DECLARE_KEY_INPUT(eF12, F12),
      HK_DECLARE_KEY_INPUT(eF13, F13),
      HK_DECLARE_KEY_INPUT(eF14, F14),
      HK_DECLARE_KEY_INPUT(eF15, F15),
      HK_DECLARE_KEY_INPUT(eF16, F16),
      HK_DECLARE_KEY_INPUT(eF17, F17),
      HK_DECLARE_KEY_INPUT(eF18, F18),
      HK_DECLARE_KEY_INPUT(eF19, F19),
      HK_DECLARE_KEY_INPUT(eF20, F20),
      HK_DECLARE_KEY_INPUT(eF21, F21),
      HK_DECLARE_KEY_INPUT(eF22, F22),
      HK_DECLARE_KEY_INPUT(eF23, F23),
      HK_DECLARE_KEY_INPUT(eF24, F24),
      HK_DECLARE_KEY_INPUT(eF25, F25),
      HK_DECLARE_KEY_INPUT(eKp_0, KP_0),
      HK_DECLARE_KEY_INPUT(eKp_1, KP_1),
      HK_DECLARE_KEY_INPUT(eKp_2, KP_2),
      HK_DECLARE_KEY_INPUT(eKp_3, KP_3),
      HK_DECLARE_KEY_INPUT(eKp_4, KP_4),
      HK_DECLARE_KEY_INPUT(eKp_5, KP_5),
      HK_DECLARE_KEY_INPUT(eKp_6, KP_6),
      HK_DECLARE_KEY_INPUT(eKp_7, KP_7),
      HK_DECLARE_KEY_INPUT(eKp_8, KP_8),
      HK_DECLARE_KEY_INPUT(eKp_9, KP_9),
      HK_DECLARE_KEY_INPUT(eKp_Decimal, KP_DECIMAL),
      HK_DECLARE_KEY_INPUT(eKp_Divide, KP_DIVIDE),
      HK_DECLARE_KEY_INPUT(eKp_Multiply, KP_MULTIPLY),
      HK_DECLARE_KEY_INPUT(eKp_Subtract, KP_SUBTRACT),
      HK_DECLARE_KEY_INPUT(eKp_Add, KP_ADD),
      HK_DECLARE_KEY_INPUT(eKp_Enter, KP_ENTER),
      HK_DECLARE_KEY_INPUT(eKp_Equal, KP_EQUAL),
      HK_DECLARE_KEY_INPUT(eLeft_Shift, LEFT_SHIFT),
      HK_DECLARE_KEY_INPUT(eLeft_Control, LEFT_CONTROL),
      HK_DECLARE_KEY_INPUT(eLeft_Alt, LEFT_ALT),
      HK_DECLARE_KEY_INPUT(eLeft_Super, LEFT_SUPER),
      HK_DECLARE_KEY_INPUT(eRight_Shift, RIGHT_SHIFT),
      HK_DECLARE_KEY_INPUT(eRight_Control, RIGHT_CONTROL),
      HK_DECLARE_KEY_INPUT(eRight_Alt, RIGHT_ALT),
      HK_DECLARE_KEY_INPUT(eRight_Super, RIGHT_SUPER),
      HK_DECLARE_KEY_INPUT(eMenu, MENU),
      eCount
    };
    enum class MouseButtonInput : U8 {
      e1, e2, e3, e4, e5, e6, e7, e8, eCount
    };
    static constexpr MouseButtonInput MouseButtonInputLast = MouseButtonInput::e8;
    static constexpr MouseButtonInput MouseButtonInputLeft = MouseButtonInput::e1;
    static constexpr MouseButtonInput MouseButtonInputRight = MouseButtonInput::e2;
    static constexpr MouseButtonInput MouseButtonInputMiddle = MouseButtonInput::e3;
    using MouseOffset2D = Offset2<F64>;
    enum class KeyModFlagBits : U8 {
      eNone = 0x0000,
      eShift = 0x0001,
      eControl = 0x0002,
      eAlt = 0x0004,
      eSuper = 0x0008,
      eCapsLock = 0x0010,
      eNumLock = 0x0020,
      eMask = 0x003F
    };
    using      KeyModFlags = Flags<KeyModFlagBits>;
    enum class PressStateFlagBits : U8 {
      eRelease = 0x00,
      ePress = 0x01,
      eUpdate = 0x02,
      eMask = 0x03
    };
    using PressStateFlags = Flags<PressStateFlagBits>;
    static inline constexpr PressStateFlags PressStateFlagsRelease = PressStateFlagBits::eRelease | PressStateFlagBits::eUpdate;
    static inline constexpr PressStateFlags PressStateFlagsPress   = PressStateFlagBits::ePress   | PressStateFlagBits::eUpdate;
    static inline constexpr PressStateFlags PressStateFlagsRepeat  = PressStateFlagBits::ePress;
  }
}
