#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
  struct Medium : public std::enable_shared_from_this<Medium> {
    virtual ~Medium();
    virtual Uuid getID() const = 0;
    template<typename DeriveType>
    auto convert() -> std::shared_ptr<DeriveType> {
      if (DeriveType::ID() == getID()) {
        return std::static_pointer_cast<DeriveType>(shared_from_this());
      }
      else {
        return nullptr;
      }
    }
  protected:
    Medium();
  };
}
