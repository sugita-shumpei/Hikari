#pragma once
#include <hikari/core/node.h>
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // CameraObject...視線を管理するためのObject
    // 
    struct CameraObject : public NodeComponentObject {
      virtual ~CameraObject() {}
    };
  }
}
