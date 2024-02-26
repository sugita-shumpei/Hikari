#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <hikari/core/data_types.h>
#include <hikari/core/flags.h>
#include <hikari/window/common.h>

#define HK_DECLARE_KEY_INPUT(KEY, GLFW_KEY) KEY
#define HK_DECLARE_CASE_CONVERT_KEY_2_INT(KEY, GLFW_KEY) \
    case KeyInput::KEY:                                  \
        return GLFW_KEY_##GLFW_KEY
#define HK_DECLARE_CASE_CONVERT_INT_2_KEY(KEY, GLFW_KEY) \
    case GLFW_KEY_##GLFW_KEY:                            \
        return KeyInput::KEY
namespace hikari
{
    enum class KeyInput:U8
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
    inline constexpr auto convertKeyInput2Int(KeyInput i) -> I32
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
        return GLFW_KEY_LAST+1;
    }
    inline constexpr auto convertInt2KeyInput(I32      i) -> KeyInput
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
    enum class KeyModFlagBits : U8 {
      eNone     = 0x0000,
      eShift    = 0x0001,
      eControl  = 0x0002,
      eAlt      = 0x0004,
      eSuper    = 0x0008,
      eCapsLock = 0x0010,
      eNumLock  = 0x0020,
      eMask     = 0x003F
    };
    template<> struct FlagsTraits<KeyModFlagBits>:std::true_type {
      using base_type = U8;
      static constexpr base_type none_mask = 0;
      static constexpr base_type all_mask  = 0x3F;
    };
    using KeyModFlags = Flags<KeyModFlagBits>;
    inline constexpr auto convertKeyMods2Int(KeyModFlags i) -> I32 {
      I32 res = 0;
      if (i & KeyModFlagBits::eShift) { res |= GLFW_MOD_SHIFT; }
      if (i & KeyModFlagBits::eControl) { res |= GLFW_MOD_CONTROL; }
      if (i & KeyModFlagBits::eAlt) { res |= GLFW_MOD_ALT; }
      if (i & KeyModFlagBits::eSuper) { res |= GLFW_MOD_SUPER; }
      if (i & KeyModFlagBits::eCapsLock) { res |= GLFW_MOD_CAPS_LOCK; }
      if (i & KeyModFlagBits::eNumLock) { res |= GLFW_MOD_NUM_LOCK; }
      return res;
    }
    inline constexpr auto convertInt2KeyMods(I32 i) -> KeyModFlags {
      KeyModFlags res = {};
      if (i & GLFW_MOD_SHIFT) { res |= KeyModFlagBits::eShift; }
      if (i & GLFW_MOD_CONTROL) { res |= KeyModFlagBits::eControl; }
      if (i & GLFW_MOD_ALT) { res |= KeyModFlagBits::eAlt; }
      if (i & GLFW_MOD_SUPER) { res |= KeyModFlagBits::eSuper; }
      if (i & GLFW_MOD_CAPS_LOCK) { res |= KeyModFlagBits::eCapsLock; }
      if (i & GLFW_MOD_NUM_LOCK) { res |= KeyModFlagBits::eNumLock; }
      return res;
    }
    enum class KeyStateFlagBits : U8 {
      eRelease = 0x00,
      ePress   = 0x01,
      eUpdate  = 0x02
    };
    template<> struct FlagsTraits<KeyStateFlagBits> :std::true_type {
      using base_type = U8;
      static constexpr base_type none_mask = 0;
      static constexpr base_type all_mask  = 0x3;
    };
    using KeyStateFlags = Flags<KeyStateFlagBits>;
    static_assert((U8)(KeyStateFlagBits::ePress  | KeyStateFlagBits::eUpdate ) == 0x03, "");
    static_assert((U8)(KeyStateFlagBits::eRelease| KeyStateFlagBits::eUpdate ) == 0x02, "");
    static_assert((U8)(KeyStateFlagBits::ePress  ) == 0x01, "");
}
