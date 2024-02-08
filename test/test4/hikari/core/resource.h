#pragma once
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    enum class ResourceType {
      eBuffer ,
      eTexture,
      eSampler
    };
    struct ResourceObject : public Object {};
    struct Resource       {};
  }
}
