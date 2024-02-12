#pragma once
#include <utility>
#include <variant>
#include <optional>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <fmt/format.h>
#include <hikari/core/data_type.h>
namespace hikari {
  inline namespace core {
    enum class TransformType { eTRS, eMat };
    struct TransformTRSData {
      Vec3 position = Vec3(0.0f);
      Quat rotation = Quat(1.0f, 0.0f, 0.0f, 0.0f);
      Vec3 scale = Vec3(1.0f);

      Bool operator==(const TransformTRSData& t)const {
        return (position == t.position) && (rotation == t.rotation) && (scale == t.scale);
      }
      Bool operator!=(const TransformTRSData& t)const {
        return !operator==(t);
      }

      auto getMat() const -> Mat4 {
        auto tran = glm::translate(position);
        auto rota = glm::toMat4(rotation);
        auto scal = glm::scale(scale);
        return tran * rota * scal;
      }
    };
    using  TransformMatData = Mat4;
    struct Transform {
      Transform() noexcept : m_data{ TransformTRSData() } {}
      Transform(const Transform& transform) = default;
      Transform& operator=(const Transform& transform) = default;

      Transform(const Vec3& t, const Quat& r, const Vec3& s = Vec3(1.0f)) noexcept : m_data{ TransformTRSData{t,r,s} } {}
      explicit Transform(const Vec3& tra) noexcept : m_data{ TransformTRSData{tra,Quat(1.0f,0.0f,0.0f,0.0f),Vec3(1.0f)} } {}
      explicit Transform(const TransformTRSData& trs) noexcept : m_data{ trs } {}
      explicit Transform(const TransformMatData& mat) noexcept : m_data{ TransformMatData() } { setMat(mat); }
      operator Mat4() const noexcept { return getMat(); }

      Bool operator==(const Transform& t)const {
        return m_data == t.m_data;
      }
      Bool operator!=(const Transform& t)const {
        return m_data != t.m_data;
      }


      auto getType() const ->TransformType { return static_cast<TransformType>(m_data.index()); }
      void setTRS(const TransformTRSData& trs) noexcept { m_data = trs; }
      void setMat(const TransformMatData& mat) noexcept {
        auto scale = glm::vec3();
        auto translate = glm::vec3();
        auto rotation = glm::quat();
        auto skew = glm::vec3();
        auto pers = glm::vec4();
        if (glm::decompose(mat, scale, rotation, translate, skew, pers)) {
          TransformTRSData trs = {};
          trs.position = translate;
          trs.rotation = rotation;
          trs.scale = scale;
          m_data = trs;
        }
        else {
          m_data = mat;
        }
      }

      Bool getTRS(TransformTRSData& trs) const noexcept {
        if (m_data.index() == (int)TransformType::eTRS) {
          trs = std::get<(int)TransformType::eTRS>(m_data);
          return true;
        }
        else {
          return false;
        }
      }
      auto getTRS() const noexcept -> std::optional<TransformTRSData> {
        if (m_data.index() == (int)TransformType::eTRS) {
          return std::get<(int)TransformType::eTRS>(m_data);
        }
        else {
          return std::nullopt;
        }
      }
      auto getMat() const noexcept  -> TransformMatData {
        if (m_data.index() == (int)TransformType::eTRS) {
          return std::get<(int)TransformType::eTRS>(m_data).getMat();
        }
        else {
          return std::get<(int)TransformType::eMat>(m_data);
        }
      }

      auto getPosition() const -> std::optional<Vec3> {
        if (m_data.index() == (int)TransformType::eTRS) {
          auto trs = std::get<(int)TransformType::eTRS>(m_data);
          return trs.position;
        }
        else {
          return std::nullopt;
        }
      }
      auto getRotation() const -> std::optional<Quat> {
        if (m_data.index() == (int)TransformType::eTRS) {
          auto trs = std::get<(int)TransformType::eTRS>(m_data);
          return trs.rotation;
        }
        else {
          return std::nullopt;
        }
      }
      auto getScale() const -> std::optional<Vec3> {
        if (m_data.index() == (int)TransformType::eTRS) {
          auto trs = std::get<(int)TransformType::eTRS>(m_data);
          return trs.scale;
        }
        else {
          return std::nullopt;
        }
      }

      Transform inverse() const {
        if (isUniformScaling()) {
          auto trs = std::get<(int)TransformType::eTRS>(m_data);
          auto inv_rot = glm::inverse(trs.rotation);
          TransformTRSData res;
          res.position = inv_rot * (-trs.position / trs.scale);
          res.rotation = inv_rot;
          res.scale = glm::vec3(1.0f) / trs.scale;
          return Transform(res);
        }
        else {
          return Transform(glm::inverse(getMat()));
        }
      }
      Bool isUniformScaling() const {
        if (m_data.index() != 0) { return false; }
        auto& trs = std::get<0>(m_data);
        return ((trs.scale.x == trs.scale.y) && (trs.scale.x == trs.scale.z));
      }
    private:
      std::variant<TransformTRSData, TransformMatData> m_data;
    };
    inline Transform operator*(const Transform& t1, const Transform& t2) {
      if ((t1.getType() == TransformType::eTRS) && (t2.getType() == TransformType::eTRS)) {
        TransformTRSData trs1; t1.getTRS(trs1);
        TransformTRSData trs2; t2.getTRS(trs2);
        if (t1.isUniformScaling()) {
          TransformTRSData trs;
          trs.position = trs1.rotation * (trs1.scale * trs2.position) + trs1.position;
          trs.rotation = trs1.rotation * trs2.rotation;
          trs.scale = trs1.scale * trs2.scale;
          return Transform(trs);
        }
        else {
          return Transform(t1.getMat() * t2.getMat());
        }
      }
      else {
        return Transform(t1.getMat() * t2.getMat());
      }
    }
    HK_TYPE_2_STRING_DEFINE(Transform);
  }
}
