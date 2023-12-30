#pragma once
#include <hikari/core/bsdf.h>
namespace hikari {
  struct BsdfNull : public Bsdf {
    static constexpr Uuid ID() { return Uuid::from_string("1B28687E-A8AA-4785-ABE4-4CB0409A5E84").value(); }
    static auto  create() -> std::shared_ptr<BsdfNull>;
    virtual ~BsdfNull();
    virtual Uuid getID() const override;
  private:
    BsdfNull();
  };
}
