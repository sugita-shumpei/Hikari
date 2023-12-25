#pragma once
#include <memory>
namespace hikari {
  struct Properties;
  struct Object : public std::enable_shared_from_this<Object> {
    virtual ~Object() noexcept;
    virtual Uuid getID() const = 0;
    
  };
}
