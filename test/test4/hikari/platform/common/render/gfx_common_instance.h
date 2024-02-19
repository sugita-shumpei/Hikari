#pragma once
#include <hikari/render/gfx_instance.h>
namespace hikari {
  inline namespace platforms {
    namespace common {
      inline namespace render {
        auto createGFXInstance(GFXAPI api) -> hikari::render::GFXInstance;
      }
    }
  }
}
