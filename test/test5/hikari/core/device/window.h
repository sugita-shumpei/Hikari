#pragma once
#include <hikari/core/types/data_types.h>
#include <hikari/core/device/system.h>
namespace hikari {
  struct Window {
    void  setPosition(IVec2 pos);
    IVec2 getPosition()const;
    void  setSize(UVec2 size);
    UVec2 getSize()const;
    UVec2 getFrameBufferSize()const;
    DVec2 getCursorPosition()const;
  private:
    IVec2 m_position;
    UVec2 m_size;
    UVec2 m_frame_buffer_size;
    DVec2 m_cursor_position;
  private:

  };
}
