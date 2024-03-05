
#include <hikari/core/window/window.h>
#include <atomic>
#include <stdexcept>
#include <hikari/core/window/app.h>
#include <hikari/core/window/renderer.h>
#include <hikari/core/window/event.h>
#include <hikari/core/input/common.h>
#include <hikari/core/input/event.h>
#include "impl_glfw_context.h"

#define HK_DECLARE_CASE_CONVERT_KEY_2_INT(KEY, GLFW_KEY) \
    case KeyInput::KEY:                                  \
        return GLFW_KEY_##GLFW_KEY
#define HK_DECLARE_CASE_CONVERT_INT_2_KEY(KEY, GLFW_KEY) \
    case GLFW_KEY_##GLFW_KEY:                            \
        return KeyInput::KEY

namespace hikari {
  namespace core {
    static constexpr auto convertKeyInput2Int(hikari::core::KeyInput i) -> I32
    {
      switch (i)
      {
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eSpace, SPACE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eApostrophe, APOSTROPHE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eComma, COMMA);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eMinus, MINUS);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(ePeriod, PERIOD);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eSlash, SLASH);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e0, 0);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e1, 1);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e2, 2);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e3, 3);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e4, 4);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e5, 5);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e6, 6);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e7, 7);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e8, 8);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(e9, 9);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eSemicolon, SEMICOLON);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eEqual, EQUAL);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eA, A);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eB, B);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eC, C);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eD, D);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eE, E);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF, F);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eG, G);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eH, H);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eI, I);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eJ, J);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eK, K);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eL, L);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eM, M);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eN, N);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eO, O);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eP, P);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eQ, Q);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eR, R);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eS, S);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eT, T);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eU, U);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eV, V);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eW, W);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eX, X);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eY, Y);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eZ, Z);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft_Bracket, LEFT_BRACKET);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eBackslash, BACKSLASH);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight_Bracket, RIGHT_BRACKET);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eGrave_Accent, GRAVE_ACCENT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eWorld_1, WORLD_1);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eWorld_2, WORLD_2);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eEscape, ESCAPE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eEnter, ENTER);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eTab, TAB);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eBackspace, BACKSPACE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eInsert, INSERT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eDelete, DELETE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight, RIGHT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft, LEFT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eDown, DOWN);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eUp, UP);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(ePage_Up, PAGE_UP);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(ePage_Down, PAGE_DOWN);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eHome, HOME);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eEnd, END);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eCaps_Lock, CAPS_LOCK);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eScroll_Lock, SCROLL_LOCK);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eNum_Lock, NUM_LOCK);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(ePrint_Screen, PRINT_SCREEN);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(ePause, PAUSE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF1, F1);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF2, F2);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF3, F3);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF4, F4);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF5, F5);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF6, F6);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF7, F7);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF8, F8);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF9, F9);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF10, F10);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF11, F11);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF12, F12);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF13, F13);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF14, F14);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF15, F15);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF16, F16);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF17, F17);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF18, F18);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF19, F19);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF20, F20);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF21, F21);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF22, F22);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF23, F23);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF24, F24);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eF25, F25);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_0, KP_0);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_1, KP_1);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_2, KP_2);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_3, KP_3);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_4, KP_4);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_5, KP_5);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_6, KP_6);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_7, KP_7);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_8, KP_8);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_9, KP_9);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Decimal, KP_DECIMAL);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Divide, KP_DIVIDE);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Multiply, KP_MULTIPLY);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Subtract, KP_SUBTRACT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Add, KP_ADD);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Enter, KP_ENTER);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eKp_Equal, KP_EQUAL);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft_Shift, LEFT_SHIFT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft_Control, LEFT_CONTROL);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft_Alt, LEFT_ALT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eLeft_Super, LEFT_SUPER);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight_Shift, RIGHT_SHIFT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight_Control, RIGHT_CONTROL);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight_Alt, RIGHT_ALT);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eRight_Super, RIGHT_SUPER);
        HK_DECLARE_CASE_CONVERT_KEY_2_INT(eMenu, MENU);
      }
      return GLFW_KEY_LAST + 1;
    }
    static constexpr auto convertInt2KeyInput(I32      i) -> KeyInput
    {
      switch (i)
      {
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eSpace, SPACE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eApostrophe, APOSTROPHE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eComma, COMMA);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eMinus, MINUS);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(ePeriod, PERIOD);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eSlash, SLASH);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e0, 0);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e1, 1);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e2, 2);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e3, 3);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e4, 4);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e5, 5);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e6, 6);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e7, 7);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e8, 8);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(e9, 9);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eSemicolon, SEMICOLON);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eEqual, EQUAL);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eA, A);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eB, B);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eC, C);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eD, D);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eE, E);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF, F);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eG, G);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eH, H);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eI, I);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eJ, J);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eK, K);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eL, L);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eM, M);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eN, N);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eO, O);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eP, P);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eQ, Q);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eR, R);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eS, S);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eT, T);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eU, U);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eV, V);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eW, W);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eX, X);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eY, Y);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eZ, Z);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft_Bracket, LEFT_BRACKET);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eBackslash, BACKSLASH);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight_Bracket, RIGHT_BRACKET);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eGrave_Accent, GRAVE_ACCENT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eWorld_1, WORLD_1);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eWorld_2, WORLD_2);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eEscape, ESCAPE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eEnter, ENTER);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eTab, TAB);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eBackspace, BACKSPACE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eInsert, INSERT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eDelete, DELETE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight, RIGHT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft, LEFT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eDown, DOWN);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eUp, UP);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(ePage_Up, PAGE_UP);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(ePage_Down, PAGE_DOWN);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eHome, HOME);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eEnd, END);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eCaps_Lock, CAPS_LOCK);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eScroll_Lock, SCROLL_LOCK);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eNum_Lock, NUM_LOCK);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(ePrint_Screen, PRINT_SCREEN);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(ePause, PAUSE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF1, F1);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF2, F2);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF3, F3);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF4, F4);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF5, F5);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF6, F6);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF7, F7);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF8, F8);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF9, F9);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF10, F10);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF11, F11);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF12, F12);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF13, F13);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF14, F14);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF15, F15);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF16, F16);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF17, F17);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF18, F18);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF19, F19);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF20, F20);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF21, F21);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF22, F22);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF23, F23);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF24, F24);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eF25, F25);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_0, KP_0);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_1, KP_1);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_2, KP_2);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_3, KP_3);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_4, KP_4);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_5, KP_5);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_6, KP_6);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_7, KP_7);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_8, KP_8);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_9, KP_9);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Decimal, KP_DECIMAL);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Divide, KP_DIVIDE);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Multiply, KP_MULTIPLY);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Subtract, KP_SUBTRACT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Add, KP_ADD);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Enter, KP_ENTER);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eKp_Equal, KP_EQUAL);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft_Shift, LEFT_SHIFT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft_Control, LEFT_CONTROL);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft_Alt, LEFT_ALT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eLeft_Super, LEFT_SUPER);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight_Shift, RIGHT_SHIFT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight_Control, RIGHT_CONTROL);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight_Alt, RIGHT_ALT);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eRight_Super, RIGHT_SUPER);
        HK_DECLARE_CASE_CONVERT_INT_2_KEY(eMenu, MENU);
      }
      return KeyInput::eCount;
    }
    static constexpr auto convertMouseButtonInput2Int(hikari::core::MouseButtonInput i) -> I32 {
      I32 res = 0;
      if (i == MouseButtonInput::e1) { return GLFW_MOUSE_BUTTON_1; }
      if (i == MouseButtonInput::e2) { return GLFW_MOUSE_BUTTON_2; }
      if (i == MouseButtonInput::e3) { return GLFW_MOUSE_BUTTON_3; }
      if (i == MouseButtonInput::e4) { return GLFW_MOUSE_BUTTON_4; }
      if (i == MouseButtonInput::e5) { return GLFW_MOUSE_BUTTON_5; }
      if (i == MouseButtonInput::e6) { return GLFW_MOUSE_BUTTON_6; }
      if (i == MouseButtonInput::e7) { return GLFW_MOUSE_BUTTON_7; }
      if (i == MouseButtonInput::e8) { return GLFW_MOUSE_BUTTON_8; }
      return res;
    }
    static constexpr auto convertInt2MouseButtonInput(I32 i) -> MouseButtonInput {
      MouseButtonInput res = {};
      if (i == GLFW_MOUSE_BUTTON_1) { return MouseButtonInput::e1; }
      if (i == GLFW_MOUSE_BUTTON_2) { return MouseButtonInput::e2; }
      if (i == GLFW_MOUSE_BUTTON_3) { return MouseButtonInput::e3; }
      if (i == GLFW_MOUSE_BUTTON_4) { return MouseButtonInput::e4; }
      if (i == GLFW_MOUSE_BUTTON_5) { return MouseButtonInput::e5; }
      if (i == GLFW_MOUSE_BUTTON_6) { return MouseButtonInput::e6; }
      if (i == GLFW_MOUSE_BUTTON_7) { return MouseButtonInput::e7; }
      if (i == GLFW_MOUSE_BUTTON_8) { return MouseButtonInput::e8; }
      return res;
    }
    static constexpr auto convertKeyMods2Int(hikari::core::KeyModFlags i) -> I32 {
      I32 res = 0;
      if (i & KeyModFlagBits::eShift) { res |= GLFW_MOD_SHIFT; }
      if (i & KeyModFlagBits::eControl) { res |= GLFW_MOD_CONTROL; }
      if (i & KeyModFlagBits::eAlt) { res |= GLFW_MOD_ALT; }
      if (i & KeyModFlagBits::eSuper) { res |= GLFW_MOD_SUPER; }
      if (i & KeyModFlagBits::eCapsLock) { res |= GLFW_MOD_CAPS_LOCK; }
      if (i & KeyModFlagBits::eNumLock) { res |= GLFW_MOD_NUM_LOCK; }
      return res;
    }
    static constexpr auto convertInt2KeyMods(I32 i) -> KeyModFlags {
      KeyModFlags res = {};
      if (i & GLFW_MOD_SHIFT) { res |= KeyModFlagBits::eShift; }
      if (i & GLFW_MOD_CONTROL) { res |= KeyModFlagBits::eControl; }
      if (i & GLFW_MOD_ALT) { res |= KeyModFlagBits::eAlt; }
      if (i & GLFW_MOD_SUPER) { res |= KeyModFlagBits::eSuper; }
      if (i & GLFW_MOD_CAPS_LOCK) { res |= KeyModFlagBits::eCapsLock; }
      if (i & GLFW_MOD_NUM_LOCK) { res |= KeyModFlagBits::eNumLock; }
      return res;
    }
    static constexpr auto convertPressState2Int(hikari::core::PressStateFlags i) -> int {
      if (i == PressStateFlagsRelease) {
        return GLFW_RELEASE;
      }
      if (i == PressStateFlagsPress) {
        return GLFW_PRESS;
      }
      if (i == PressStateFlagsRepeat) {
        return GLFW_REPEAT;
      }
      return  GLFW_REPEAT;
    }
    static constexpr auto convertInt2PressState(int i) -> PressStateFlags {
      if (i == GLFW_PRESS) {
        return PressStateFlagsPress;
      }
      if (i == GLFW_RELEASE) {
        return PressStateFlagsRelease;
      }
      if (i == GLFW_REPEAT) {
        return PressStateFlagsRepeat;
      }
      return  PressStateFlagsRepeat;
    }

  }
}

static auto convertHandle2WindowData(GLFWwindow* handle) -> hikari::core::WindowData* {
  return reinterpret_cast<hikari::core::WindowData*>(glfwGetWindowUserPointer(handle));
}


hikari::core::Window::Window(WindowApp* app, const WindowDesc& desc)
  :m_app{ app }, m_handle{ nullptr }, m_data{ this, desc.size,{},{},desc.position,{},desc.title }
{
  auto& instance = hikari::core::GLFWContext::getInstance();
  instance.addRef();
  m_data.is_resizable = (desc.flags & hikari::core::WindowFlagBits::eResizable);
  m_data.is_visible   = (desc.flags & hikari::core::WindowFlagBits::eVisible  );
  m_data.is_floating  = (desc.flags & hikari::core::WindowFlagBits::eFloating );
  m_data.is_graphics_opengl = (desc.flags & hikari::core::WindowFlagBits::eGraphicsOpenGL);
  m_data.is_graphics_vulkan = (desc.flags & hikari::core::WindowFlagBits::eGraphicsVulkan);
  if (m_data.is_graphics_opengl && m_data.is_graphics_vulkan) {
    instance.release();
    throw std::runtime_error("[hikari::core::Window::Window(WindowApp* app, const WindowDesc& desc)] Graphics API Must Be Single!");
  }
  if (m_data.is_graphics_opengl) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  }
  else {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  glfwWindowHint(GLFW_RESIZABLE, m_data.is_resizable);
  glfwWindowHint(GLFW_VISIBLE  , m_data.is_visible);
  glfwWindowHint(GLFW_FLOATING , m_data.is_floating);
  //glfwWindowHint(GLFW_FLOATING , m_data.is_floating);
  GLFWwindow* handle = glfwCreateWindow(desc.size.width,desc.size.height,desc.title.c_str(),nullptr,nullptr);
  glfwDefaultWindowHints();
  if (!handle) {
    instance.release();
    throw std::runtime_error("[hikari::core::Window::Window(WindowApp* app, const WindowDesc& desc)] Failed To Create GLFW Window!");
  }
  glfwMakeContextCurrent(handle);
  m_handle = handle;
  glfwSetWindowPos(handle, desc.position.x, desc.position.y);
  int fb_size_w = 0; int fb_size_h = 0;
  glfwGetFramebufferSize(handle, &fb_size_w, &fb_size_h);
  m_data.surface_size = {(uint32_t) fb_size_w,(uint32_t) fb_size_h };
  glfwSetWindowUserPointer(handle, &this->m_data);
  glfwSetWindowSizeCallback(handle, [](auto handle_,I32 w, I32 h) {
    auto data = convertHandle2WindowData(handle_);
    if (data->is_fullscreen) { return; }
    data->size.width  = w;
    data->size.height = h;
    // eventを起動する
    auto& event_manager =data->window->getWindowEventManager();
    auto event = new WindowResizeEvent();
    event->window = data->window;
    event->size   = data->size;
    event_manager.submitEvent(std::unique_ptr<IEvent>(event));
  });
  glfwSetFramebufferSizeCallback(handle, [](auto handle_,I32 w, I32 h) {
    auto data = convertHandle2WindowData(handle_);
    data->surface_size.width  = w;
    data->surface_size.height = h;
  });
  glfwSetWindowPosCallback(handle, [](auto handle_, I32 x, I32 y) {
    auto data = convertHandle2WindowData(handle_);
    if (data->is_fullscreen) { return; }
    data->position.x = x;
    data->position.y = y;
    // eventを起動する
    auto& event_manager = data->window->getWindowEventManager();
    auto event = new WindowMovedEvent();
    event->window = data->window;
    event->position = data->position;
    event_manager.submitEvent(std::unique_ptr<IEvent>(event));
    });
  glfwSetWindowCloseCallback(handle, [](auto handle_) {
    auto data = convertHandle2WindowData(handle_);
    data->is_closed = true;
    auto& event_manager = data->window->getWindowEventManager();
    auto event = new WindowCloseEvent();
    event->window = data->window;
    event_manager.submitEvent(std::unique_ptr<IEvent>(event));
  });
  glfwSetWindowFocusCallback(handle, [](auto handle_,int is_focused) {
    auto data = convertHandle2WindowData(handle_);
    data->is_focused = is_focused;
    auto& event_manager = data->window->getWindowEventManager();
    if (is_focused) {
      auto event = new WindowFocusEvent();
      event->window = data->window;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
    }
    else {
      auto event = new WindowLeaveEvent();
      event->window = data->window;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
    }
  });
  glfwSetWindowIconifyCallback(handle, [](auto handle_, int is_iconified) {
    auto data = convertHandle2WindowData(handle_);
    data->is_iconified = is_iconified;
    auto& event_manager = data->window->getWindowEventManager();
    if (is_iconified) {
      auto event = new WindowIconifiedEvent();
      event->window = data->window;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
    }
    else {
      auto event = new WindowRestoredEvent();
      event->window = data->window;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
    }
  });
  glfwSetKeyCallback(handle, [](auto handle_, int key, int scancode, int state, int mods) {
    auto data      = convertHandle2WindowData(handle_);
    auto key_input = hikari::core::convertInt2KeyInput(key);
    auto key_state = hikari::core::convertInt2PressState(state);
    auto key_mods  = hikari::core::convertInt2KeyMods(mods);
    auto& event_manager = data->window->getInputEventManager();
    data->input.setKey(key_input, key_mods,key_state);
    if (key_state == hikari::core::PressStateFlagsPress) {
      auto event = new KeyPressedEvent();
      event->key = key_input;
      event->mod = key_mods;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
      return;
    }
    if (key_state == hikari::core::PressStateFlagsRelease) {
      auto event = new KeyReleasedEvent();
      event->key = key_input;
      event->mod = key_mods;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
      return;
    }
    if (key_state == hikari::core::PressStateFlagsRepeat) {
      auto event = new KeyRepeatedEvent();
      event->key = key_input;
      event->mod = key_mods;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
      return;
    }
  });
  glfwSetMouseButtonCallback(handle, [](auto handle_, int mouse, int state, int mods) {
    auto data = convertHandle2WindowData(handle_);
    auto mouse_input = hikari::core::convertInt2MouseButtonInput(mouse);
    auto key_state = hikari::core::convertInt2PressState(state);
    auto key_mods = hikari::core::convertInt2KeyMods(mods);
    auto& event_manager = data->window->getInputEventManager();
    data->input.setMouseButton(mouse_input, key_mods, key_state);
    if (key_state == hikari::core::PressStateFlagsPress) {
      auto event = new MouseButtonPressedEvent();
      event->button = mouse_input;
      event->mod = key_mods;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
      return;
    }
    if (key_state == hikari::core::PressStateFlagsRelease) {
      auto event = new MouseButtonReleasedEvent();
      event->button = mouse_input;
      event->mod = key_mods;
      event_manager.submitEvent(std::unique_ptr<IEvent>(event));
      return;
    }
  });
  glfwSetCursorPosCallback(handle, [](auto handle_, double x, double y) {
    auto data = convertHandle2WindowData(handle_);
    data->input.setMousePosition(MouseOffset2D(x,y));
  });
  glfwSetCursorEnterCallback(handle, [](auto handle_, int enter) {
    auto data = convertHandle2WindowData(handle_);
    data->is_hovered = enter;
  });
  glfwSetScrollCallback(handle, [](auto handle_, double x, double y) {
    auto data = convertHandle2WindowData(handle_);
    data->input.setMouseScrollOffset(MouseOffset2D(x,y));
  });
}

hikari::core::Window::~Window() noexcept
{
  if (m_renderer) {
    m_renderer->terminate();
  }
  if (!m_handle) { return; }
  glfwDestroyWindow((GLFWwindow*)m_handle);
  auto& instance = hikari::core::GLFWContext::getInstance();
  instance.release();
}

//

auto hikari::core::Window::getApp() -> WindowApp* { return m_app; }
void hikari::core::Window::pollEvents()
{
  glfwPollEvents();
}
void hikari::core::Window::onSync()
{
  auto renderer = getRenderer();
  if (renderer)
    renderer->onSync();
}
auto hikari::core::Window::getSize() const -> WindowExtent2D
{
  return m_data.is_fullscreen?m_data.fullscreen_size :m_data.size;
}
void hikari::core::Window::setSize(const WindowExtent2D& size)
{
  if (m_data.is_fullscreen) { return; }
  m_data.size = size;
  glfwSetWindowSize((GLFWwindow*)m_handle, size.width, size.height);
}
auto hikari::core::Window::getSurfaceSize() const -> WindowExtent2D
{
  return m_data.surface_size;
}
auto hikari::core::Window::getPosition() const -> WindowOffset2D
{
  return m_data.is_fullscreen ? WindowOffset2D{0,0} : m_data.position;
}
void hikari::core::Window::setPosition(const WindowOffset2D& pos)
{
  if (m_data.is_fullscreen) { return; }
  m_data.position = pos;
  glfwSetWindowPos((GLFWwindow*)m_handle, pos.x, pos.y);
}
auto hikari::core::Window::getNativeHandle() -> void*
{
  return m_handle;
}
auto hikari::core::Window::getTitle() const -> const std::string&
{
  return m_data.title;
}
void hikari::core::Window::setTitle(const std::string& title)
{
  if (m_data.is_iconified) { return; }
  glfwSetWindowTitle((GLFWwindow*)m_handle, title.c_str());
  m_data.title = title;
}
void hikari::core::Window::setFullscreen(bool is_fullscreen)
{
  if (is_fullscreen) {
    if (m_data.is_fullscreen) { return; }
    auto monitor = glfwGetPrimaryMonitor();
    if (m_data.fullscreen_size.width == 0) {
      auto videocode = glfwGetVideoMode(monitor);
      m_data.fullscreen_size = {(uint32_t)videocode->width,(uint32_t)videocode->height};
    }
    m_data.is_fullscreen = true;
    glfwSetWindowMonitor((GLFWwindow*)m_handle, monitor, 0, 0, m_data.fullscreen_size.width, m_data.fullscreen_size.height, GLFW_DONT_CARE);
    auto event = new WindowFullscreenEvent();
    auto& event_manager = getWindowEventManager();
    event->size = m_data.fullscreen_size;
    event_manager.submitEvent(std::unique_ptr<IEvent>(event));
  }
  else {
    if (!m_data.is_fullscreen) { return; }
    glfwSetWindowMonitor((GLFWwindow*)m_handle, nullptr, m_data.position.x, m_data.position.y, m_data.size.width, m_data.size.height, 0);
    m_data.is_fullscreen = false;
    auto event = new WindowWindowedEvent();
    auto& event_manager = getWindowEventManager();
    event->size = m_data.size;
    event->position = m_data.position;
    event_manager.submitEvent(std::unique_ptr<IEvent>(event));
  }
}

void hikari::core::Window::setRenderer(std::unique_ptr<WindowRenderer>&& renderer) {
  if (m_renderer) {
    m_renderer->terminate();
  }
  m_renderer = std::move(renderer);
  if (m_renderer) {
    if (!m_renderer->initialize()) {
      throw std::runtime_error("[hikari::core::Window::setRenderer(std::unique_ptr<WindowRenderer>&& renderer)] Failed To Set Renderer!");
    }
  }
}

bool hikari::core::Window::isClosed() const
{
  return m_data.is_closed;
}
bool hikari::core::Window::isFocused() const
{
  return m_data.is_focused;
}
bool hikari::core::Window::isHovered() const
{
  return m_data.is_hovered;
}
bool hikari::core::Window::isIconified() const
{
  return m_data.is_iconified;
}
bool hikari::core::Window::isFullscreen() const
{
  return m_data.is_fullscreen;
}
bool hikari::core::Window::isVisible() const
{
  return m_data.is_visible;
}
bool hikari::core::Window::isResizable() const
{
  return m_data.is_resizable;
}
bool hikari::core::Window::isFloating() const
{
  return m_data.is_floating;
}
bool hikari::core::Window::isGraphicsOpenGL() const
{
  return m_data.is_graphics_opengl;
}
bool hikari::core::Window::isGraphicsVulkan() const
{
  return m_data.is_graphics_vulkan;
}
auto hikari::core::Window::getInput() const noexcept -> const Input&
{
  return m_data.input;
}
void hikari::core::Window::updateInput()
{
  m_data.input.resetKey();
  m_data.input.resetMouse();
}
auto hikari::core::Window::getWindowEventManager() noexcept -> EventManager&
{
  return m_app->getWindowEventManager();
}
auto hikari::core::Window::getInputEventManager() noexcept -> EventManager&
{
  return m_app->getInputEventManager();
}

