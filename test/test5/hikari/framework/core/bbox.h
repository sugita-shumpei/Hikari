#pragma once
#include <glm/gtx/component_wise.hpp>
#include <hikari/framework/core/data_type.h>
namespace hikari {
  namespace core {
    template<typename VecT>
    struct BBoxT {
      BBoxT() noexcept : min{ +FLT_MAX },  max{ -FLT_MAX } {}
      BBoxT(const VecT& p0) noexcept :min{ p0 }, max{ p0 } {}
      BBoxT(const VecT& p0, const VecT& p1) noexcept :min{ glm::min(p0,p1) }, max{ glm::max(p0,p1) } {}
      BBoxT(const Array<VecT>& points) noexcept :BBoxT() {
        for (auto& p : points) {
          min = glm::min(min, p);
          max = glm::max(max, p);
        }
      }

      BBoxT(const BBoxT& bbox)noexcept :min{ bbox.min }, max{ bbox.max } {}
      BBoxT& operator=(const BBoxT& bbox)noexcept { if (this != &bbox) { min = bbox.min; max = bbox.max; } return *this; }

      Bool operator==(const BBoxT& v1)const noexcept {
        return (min == v1.min) && (max == v1.max);
      }
      Bool operator!=(const BBoxT& v1)const noexcept {
        return (min != v1.min) || (max != v1.max);
      }

      //BBoxT operator|(const BBoxT& v1)const noexcept {
      //  BBoxT b;
      //  b.min = glm::min(min, v1.min);
      //  b.max = glm::max(max, v1.max);
      //  return b;
      //}
      //BBoxT operator^(const BBoxT& v1)const noexcept {
      //  BBoxT b;
      //  auto min_ = glm::max(min, v1.min);// 
      //  auto max_ = glm::min(max, v1.max);
      //  if (!glm::any(glm::lessThan(max_ - min_, VecT(0.0f)))) {
      //    b.min = min_; b.max = max_;
      //  }
      //  return b;
      //}

      auto getMin() const noexcept -> VecT { return min; }
      void setMin(const VecT& min_) noexcept { min = glm::min(min_, min); }
      auto getMax() const noexcept -> VecT { return max; }
      void setMax(const VecT& max_) noexcept { max = glm::max(max_, max); }
      auto getRange() const noexcept -> VecT { return 0.5f * (max - min); }
      void setRange(const VecT& v) {
        auto center = 0.5f * (max + min);
        min = center - v * 0.5f;
        max = center + v * 0.5f;
      }
      auto getCenter() const noexcept -> VecT { return 0.5f * (max + min); }
      void setCenter(const VecT& v) noexcept {
        auto range = max - min;
        min = v - 0.5f * range;
        max = v + 0.5f * range;
      }
      auto getVolume() const noexcept -> F32 {
        auto r = glm::max(getRange(),VecT(0.0f));
        return glm::compMul(r);
      }
      void addPoint(const VecT& p) noexcept {
        min = glm::min(min, p);
        max = glm::max(max, p);
      }

      //Bool isValid()const noexcept {
      //  return !glm::any(glm::lessThan(getRange(), VecT(0.0f))));
      //}

      VecT min = VecT(+FLT_MAX);
      VecT max = VecT(-FLT_MAX);
    };
    using BBox2 = BBoxT<Vec2>;
    using BBox3 = BBoxT<Vec3>;
    using BBox4 = BBoxT<Vec4>;
  }
}
