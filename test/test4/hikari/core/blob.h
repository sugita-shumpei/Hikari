#pragma once
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    struct BlobObject : public Object {
      virtual ~BlobObject() noexcept;
      virtual auto getBuffer()     const -> const Byte* = 0;
      virtual auto getBufferSize() const -> const U64   = 0;
    };
    struct Blob {

    };
  }
}
