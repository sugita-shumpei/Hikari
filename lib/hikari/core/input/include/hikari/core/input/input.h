#pragma once
#include <utility>
#include <array>
#include <hikari/core/input/common.h>
namespace hikari {
  namespace core {
    struct Input {
      bool getKeyPressed(KeyInput key) const  { return m_keys[(U32)key].second == PressStateFlagsPress; }
      bool getKeyReleased(KeyInput key) const { return m_keys[(U32)key].second == PressStateFlagsRelease; }
      bool getKeyRepeated(KeyInput key) const { return m_keys[(U32)key].second == PressStateFlagsRepeat; }
      bool getKeyPressed (KeyInput key, KeyModFlags& mods) const {
        auto& pair= m_keys[(U32)key];
        bool res = (pair.second == PressStateFlagsPress);
        if (res) { mods = pair.first; return true; }
        return false;
      }
      bool getKeyReleased(KeyInput key, KeyModFlags& mods) const {
        auto& pair = m_keys[(U32)key];
        bool res = (pair.second == PressStateFlagsRelease);
        if (res) { mods = pair.first; return true; }
        return false;
      }
      bool getKeyRepeated(KeyInput key, KeyModFlags& mods) const {
        auto& pair = m_keys[(U32)key];
        bool res = (pair.second == PressStateFlagsRepeat);
        if (res) { mods = pair.first; return true; }
        return false;
      }
      bool getMousePressed (MouseButtonInput button, KeyModFlags& mods) const {
        auto& pair = m_keys[(U32)button];
        bool res = (pair.second == PressStateFlagsPress);
        if (res) { mods = pair.first; return true; }
        return false;
      }
      bool getMouseReleased(MouseButtonInput button, KeyModFlags& mods) const {
        auto& pair = m_keys[(U32)button];
        bool res = (pair.second == PressStateFlagsRelease);
        if (res) { mods = pair.first; return true; }
        return false;
      }
      auto getMousePosition() const->MouseOffset2D { return m_mouse_position; }
      auto getMouseScrollOffset() const->MouseOffset2D { return m_mouse_scroll_offset; }
    private:
      friend class Window;
      void setKey(KeyInput key, KeyModFlags mods, PressStateFlags press) {
        m_keys[(U32)key] = { mods,press };
      }
      void setMouseButton(MouseButtonInput button, KeyModFlags mods, PressStateFlags press) {
        m_mouse_buttons[(U32)button] = { mods,press };
      }
      void setMousePosition(const MouseOffset2D& mouse_position) { m_mouse_position = mouse_position; }
      void setMouseScrollOffset(const MouseOffset2D& mouse_scroll_offset) { m_mouse_scroll_offset = mouse_scroll_offset; }
      void resetKey() {
        m_keys = {};
      }
      void resetMouse() {
        m_mouse_buttons = {};
      }
    private:
      std::array<std::pair<KeyModFlags, PressStateFlags>, (U32)KeyInput::eCount> m_keys = {};
      std::array<std::pair<KeyModFlags, PressStateFlags>, (U32)MouseButtonInput::eCount> m_mouse_buttons = {};
      MouseOffset2D m_mouse_position = {};
      MouseOffset2D m_mouse_scroll_offset = {};
    };
  }
}
