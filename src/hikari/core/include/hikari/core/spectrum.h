#pragma once
#include <hikari/core/data_type.h>
#include <memory>
namespace hikari {
  struct Spectrum : public std::enable_shared_from_this<Spectrum> {
    virtual   ~Spectrum() noexcept {}
    virtual Uuid  getID() const =  0;

    template<typename DeriveType>
    auto convert() -> std::shared_ptr<DeriveType> {
      if (DeriveType::ID() == getID()) {
        return std::static_pointer_cast<DeriveType>(shared_from_this());
      }
      else {
        return nullptr;
      }
    }
  };
  using  SpectrumPtr = std::shared_ptr<Spectrum>;
}
