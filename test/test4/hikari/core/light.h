#pragma once
#include <hikari/core/node.h>
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // LightObject...光源を管理するためのObject
    // 
    struct LightObject : public Object {
      virtual ~LightObject() {}
    };
  }
}
