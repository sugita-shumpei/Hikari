#pragma once
#include <hikari/graphics/common.h>
namespace hikari {
  // GraphicsAPIのエントリーポイント
  struct GraphicsInstance {
    virtual ~GraphicsInstance()noexcept {}
    virtual auto getAPIType() const->GraphicsAPIType = 0;
    virtual bool isInitialized()const = 0;
  protected:
    friend class GraphicsSystem;
    virtual bool initialize() = 0;
    virtual void terminate() = 0;
  };
}
