#pragma once

#include <hikari/core/bsdf.h>
#include <hikari/core/variant.h>
namespace hikari {
  struct BsdfTwoSided : public Bsdf
  {
    static constexpr Uuid ID() { return Uuid::from_string("43B8E67F-3D13-4884-A8B8-1B5BBB1E44B5").value(); }
    static auto create() -> std::shared_ptr<BsdfTwoSided>;
    virtual ~BsdfTwoSided();

    auto getBsdf() -> BsdfPtr;
    void setBsdf(const BsdfPtr& bsdf);

    auto getBsdfs() ->  std::array<BsdfPtr, 2>;
    void setBsdfs(const std::array<BsdfPtr, 2>& bsdf);

    bool isSeparate() const;

    // Bsdf を介して継承されました
    Uuid getID() const override;
  private:
    BsdfTwoSided();
  private:
    std::array<BsdfPtr, 2> m_bsdfs;
  };
}
