#pragma once
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
namespace hikari {
  namespace core {
    namespace imgui_utils {
      inline void clone_draw_data(const ImDrawData& src, ImDrawData& dst)
      {
        if (!dst.CmdLists.empty()) {
          dst.CmdLists.clear_delete();
          dst.CmdListsCount = 0;
        }
        dst.Clear();
        dst = src;
        dst.CmdLists.resize(src.CmdListsCount);
        for (size_t i = 0; i < src.CmdListsCount; ++i) {
          dst.CmdLists[i] = src.CmdLists[i]->CloneOutput();
        }
      }
      inline  void  free_draw_data(ImDrawData& dst) {
        dst.CmdLists.clear_delete();
        dst.CmdListsCount = 0;
      }
    }
  }
}
