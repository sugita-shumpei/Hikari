#pragma once
#include <hikari/core/data_type.h>
#include <variant>
#include <optional>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
namespace hikari   {
  enum class TransformType { eTRS, eMat};
  struct TransformTRSData {
    Vec3 position = Vec3(0.0f);
    Quat rotation = Quat(1.0f,0.0f,0.0f,0.0f);
    Vec3 scale    = Vec3(1.0f);

    auto getMat() const -> Mat4x4 {
      auto tran = glm::translate(position);
      auto rota = glm::toMat4(rotation);
      auto scal = glm::scale(scale);
      return tran * rota * scal;
    }
  };
  using  TransformMatData = Mat4x4;
  struct Transform {
    Transform() noexcept : m_data{ TransformTRSData() } {}
    Transform(const Transform& transform) = default;
    Transform& operator=(const Transform& transform) = default;

    Transform(const TransformTRSData& trs) noexcept : m_data{ trs } {}
    Transform(const TransformMatData& mat) noexcept : m_data{ TransformTRSData() } { setMat(mat); }

    Transform& operator=(const TransformTRSData& trs) noexcept { m_data = trs; return  *this; }
    Transform& operator=(const TransformMatData& mat) noexcept { m_data = mat; return  *this; }

    Transform inverse() const {
      if (isUniformScaling()) {
        auto& trs    = std::get<0>(m_data);
        auto inv_rot = glm::inverse(trs.rotation);
        TransformTRSData res;
        res.position =  inv_rot * (-trs.position / trs.scale);
        res.rotation =  inv_rot;
        res.scale    = glm::vec3(1.0f) / trs.scale;
        return res;
      }
      else {
        return glm::inverse(getMat());
      }
    }

    auto      getType() const ->TransformType { return static_cast<TransformType>(m_data.index()); }

    void      setTRS(const TransformTRSData& trs) noexcept { m_data = trs; }
    void      setMat(const TransformMatData& mat) noexcept {
      auto scale     = glm::vec3();
      auto translate = glm::vec3();
      auto rotation  = glm::quat();
      auto skew      = glm::vec3();
      auto pers      = glm::vec4();
      if (glm::decompose(mat, scale,rotation, translate, skew, pers)) {
        TransformTRSData trs = {};
        trs.position = translate;
        trs.rotation = rotation;
        trs.scale    = scale;
        m_data = trs;
      }
      else {
        m_data = mat;
      }
    }

    Bool      getTRS(TransformTRSData& trs) const noexcept{
      if (m_data.index() == 0) {
        trs = std::get<0>(m_data);
        return true;
      }
      else {
        return false;
      }
    }
    auto      getMat() const noexcept  -> TransformMatData {
      if (m_data.index() == 0) {
        return std::get<0>(m_data).getMat();
      }
      else {
        return std::get<1>(m_data);
      }
    }

    auto getPosition() const -> std::optional<Vec3> {
      if (m_data.index() == 0) {
        auto trs = std::get<0>(m_data);
        return trs.position;
      }
      else {
        return std::nullopt;
      }
    }
    auto getRotation() const -> std::optional<Quat> {
      if (m_data.index() == 0) {
        auto trs = std::get<0>(m_data);
        return trs.rotation;
      }
      else {
        return std::nullopt;
      }
    }
    auto getScale() const -> std::optional<Vec3> {
      if (m_data.index() == 0) {
        auto trs = std::get<0>(m_data);
        return trs.scale;
      }
      else {
        return std::nullopt;
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
        trs.position = trs1.rotation *(trs1.scale * trs2.position) + trs1.position;
        trs.rotation = trs1.rotation * trs2.rotation;
        trs.scale    = trs1.scale    * trs2.scale;
        return trs;
      }
      else {
        return Transform(t1.getMat() * t2.getMat());
      }
    }
    else {
      return Transform(t1.getMat() * t2.getMat());
    }
  }
}
