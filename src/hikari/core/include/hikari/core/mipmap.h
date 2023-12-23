#pragma once
#include <hikari/core/bitmap.h>
#include <hikari/core/data_type.h>
#include <memory>
#include <vector>
namespace hikari {
    struct Mipmap {
      auto getWidth()      const -> U32;// Width
      auto getHeight()     const -> U32;// Height
      auto getDepth ()     const -> U32;// Depth
      auto getLayerCount() const -> U32;// Layer
      auto getLevelCount() const -> U32;// Level
      auto getLevel(U32 idx)->BitmapPtr;// Bitmap
    private:
      std::vector<BitmapPtr> m_levels;
    };
    using  MipmapPtr = std::shared_ptr<Mipmap>;
}
