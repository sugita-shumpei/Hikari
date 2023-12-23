#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
    struct Bitmap {
      auto getWidth()      const->U32;// Width 
      auto getHeight()     const->U32;// Height
      auto getDepth()      const->U32;// Depth 
      auto getLayerCount() const->U32;// Layer 
    private:
      U32 m_width;
      U32 m_height;
      U32 m_depth;
      U32 m_layer_count;
    };
    using BitmapPtr = std::shared_ptr<Bitmap>;
}
