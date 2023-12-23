#pragma once
#include <memory>
#include <hikari/core/data_type.h>
namespace hikari {
  enum class BsdfDistributionType {
    eBeckman,
    eGGX
  };
  struct Bsdf : public std::enable_shared_from_this<Bsdf> {
    virtual ~Bsdf();
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
    Bsdf();
  };
  using BsdfPtr = std::shared_ptr<Bsdf>;
}
