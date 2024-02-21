#pragma once
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <glm/gtx/matrix_decompose.hpp>
#include <hikari/core/types/data_type.h>
#include <hikari/core/types/vector.h>
#include <hikari/core/types/matrix.h>
#include <hikari/core/types/quaternion.h>
#include <array>
#endif
#if defined(__cplusplus)
namespace hikari {
  inline namespace core {
#endif

    enum class TransformType { eTRS, eMat };
    struct TransformTRSData {
      Vec3 position = Vec3(0.0f);
      Quat rotation = Quat(1.0f, 0.0f, 0.0f, 0.0f);
      Vec3 scaling = Vec3(1.0f);

      Bool operator==(const TransformTRSData& t)const {
        return (position == t.position) && (rotation == t.rotation) && (scaling == t.scaling);
      }
      Bool operator!=(const TransformTRSData& t)const {
        return !operator==(t);
      }

      auto getMat() const -> Mat4 {
        auto tran = glm::translate(position);
        auto rota = glm::toMat4(rotation);
        auto scal = glm::scale(scaling);
        return tran * rota * scal;
      }
    };
    using  TransformMatData = Matrix;//16
    struct Transform {
      // もし, TransformTRSを使用する場合
      // 余ったメモリ領域をすべて非正規化数
      // で初期化する
      Transform() noexcept {
        m_data[0]  = 0.0f;
        m_data[1]  = 0.0f;
        m_data[2]  = 0.0f;
        m_data[3]  = 0.0f;
        m_data[4]  = 0.0f;
        m_data[5]  = 0.0f;
        m_data[6]  = 1.0f;
        m_data[7]  = 1.0f;
        m_data[8]  = 1.0f;
        m_data[9]  = 1.0f;
        m_data[10] = FLT_MIN;
        m_data[11] = FLT_MIN;
        m_data[12] = FLT_MIN;
        m_data[13] = FLT_MIN;
        m_data[14] = FLT_MIN;
        m_data[15] = FLT_MIN;
      }
      Transform(const Transform& transform) = default;
      Transform& operator=(const Transform& transform) = default;
      Transform(const TransformTRSData& trs) noexcept {
        m_data[0]  = trs.position[0];
        m_data[1]  = trs.position[1];
        m_data[2]  = trs.position[2];
        m_data[3]  = trs.rotation[0];
        m_data[4]  = trs.rotation[1];
        m_data[5]  = trs.rotation[2];
        m_data[6]  = trs.rotation[3];
        m_data[7]  = trs.scaling[0];
        m_data[8]  = trs.scaling[1];
        m_data[9]  = trs.scaling[2];
        m_data[10] = FLT_MIN;
        m_data[11] = FLT_MIN;
        m_data[12] = FLT_MIN;
        m_data[13] = FLT_MIN;
        m_data[14] = FLT_MIN;
        m_data[15] = FLT_MIN;
      }
      Transform(const TransformMatData& mat) noexcept : m_data{} {
        setMat(mat);
      }
      Transform(const Vec3& t, const Quat& r, const Vec3& s = Vec3(1.0f)) noexcept : m_data{
        t[0],t[1],t[2],
        r[0],r[1],r[2],r[3],
        s[0],s[1],s[2],
        FLT_MIN,FLT_MIN,FLT_MIN,
        FLT_MIN,FLT_MIN,FLT_MIN
      } {}
      explicit Transform(const Vec3& tra) noexcept : Transform(tra, Quat(1.0f, 0.0f, 0.0f, 0.0f), Vec3(1.0f)) {}
      operator Mat4() const noexcept { return getMat(); }

      Bool operator==(const Transform& t)const {
        return m_data == t.m_data;
      }
      Bool operator!=(const Transform& t)const {
        return m_data != t.m_data;
      }
      auto getType() const ->TransformType {
        if (m_data[10] != FLT_MIN) { return TransformType::eMat; }
        if (m_data[11] != FLT_MIN) { return TransformType::eMat; }
        if (m_data[12] != FLT_MIN) { return TransformType::eMat; }
        if (m_data[13] != FLT_MIN) { return TransformType::eMat; }
        if (m_data[14] != FLT_MIN) { return TransformType::eMat; }
        if (m_data[15] != FLT_MIN) { return TransformType::eMat; }
        return TransformType::eTRS;
      }
      void setTRS(const TransformTRSData& trs) noexcept {
        m_data[0] = trs.position[0];
        m_data[1] = trs.position[1];
        m_data[2] = trs.position[2];
        m_data[3] = trs.rotation[0];
        m_data[4] = trs.rotation[1];
        m_data[5] = trs.rotation[2];
        m_data[6] = trs.rotation[3];
        m_data[7] = trs.scaling[0];
        m_data[8] = trs.scaling[1];
        m_data[9] = trs.scaling[2];
        m_data[10] = FLT_MIN;
        m_data[11] = FLT_MIN;
        m_data[12] = FLT_MIN;
        m_data[13] = FLT_MIN;
        m_data[14] = FLT_MIN;
        m_data[15] = FLT_MIN;
      }
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
          trs.scaling  = scale;
          setTRS(trs);
        }
        else {
          setMat(mat);
        }
      }

      Bool getTRS(TransformTRSData& trs) const noexcept {
        if (getType() == TransformType::eTRS) {
          trs.position = Vec3(m_data[0], m_data[1], m_data[2]);
          trs.rotation = Quat(m_data[6], m_data[3], m_data[4], m_data[5]);
          trs.scaling  = Vec3(m_data[7], m_data[8], m_data[9]);
          return true;
        }
        else {
          return false;
        }
      }
      auto getTRS() const noexcept -> std::optional<TransformTRSData> {
        TransformTRSData trs;
        if (getTRS(trs)) {
          return trs;
        }
        else {
          return std::nullopt;
        }
      }
      auto getMat() const noexcept  -> TransformMatData {
        if (getType() == TransformType::eTRS) {
          TransformTRSData trs;
          trs.position = Vec3(m_data[0], m_data[1], m_data[2]);
          trs.rotation = Quat(m_data[6], m_data[3], m_data[4], m_data[5]);
          trs.scaling  = Vec3(m_data[7], m_data[8], m_data[9]);
          return trs.getMat();
        }
        else {
          TransformMatData m;
          std::memcpy(&m, &m_data, sizeof(m));
          return m;
        }
      }

      auto getPosition() const -> std::optional<Vec3> {
        if (getType() == TransformType::eTRS) {
          TransformTRSData trs;
          return Vec3(m_data[0], m_data[1], m_data[2]);
        }
        else {
          return std::nullopt;
        }
      }
      auto getRotation() const -> std::optional<Quat> {
        if (getType() == TransformType::eTRS) {
          return Quat(m_data[6], m_data[3], m_data[4], m_data[5]);
        }
        else {
          return std::nullopt;
        }
      }
      auto getScale() const -> std::optional<Vec3> {
        if (getType() == TransformType::eTRS) {
          auto trs = std::get<(int)TransformType::eTRS>(m_data);
          return Vec3(m_data[7], m_data[8], m_data[9]);
        }
        else {
          return std::nullopt;
        }
      }

      Transform inverse() const {
        TransformTRSData trs;
        bool is_uniform = true;
        if (!getTRS(trs)) { is_uniform = false; }
        if (is_uniform) {
            is_uniform  = ((trs.scaling.x == trs.scaling.y) && (trs.scaling.x == trs.scaling.z));
        }
        if (is_uniform) {
          auto inv_rot = glm::inverse(trs.rotation);
          TransformTRSData res;
          res.position = inv_rot * (-trs.position / trs.scaling);
          res.rotation = inv_rot;
          res.scaling  = glm::vec3(1.0f) / trs.scaling;
          return Transform(res);
        }
        else {
          return Transform(glm::inverse(getMat()));
        }
      }
      Bool isUniformScaling() const {
        TransformTRSData trs;
        if (!getTRS(trs)) { return false; }
        return ((trs.scaling.x == trs.scaling.y) && (trs.scaling.x == trs.scaling.z));
      }

      static auto fromTranslate(const Vec3& t) noexcept -> Transform {
        return Transform(t, glm::identity<Quat>(), Vec3(1.0f));
      }
      static auto fromRotation(const Quat& r) noexcept -> Transform {
        return Transform(Vec3(0.0f), r, Vec3(1.0f));
      }
      static auto fromScale(const Quat& r) noexcept -> Transform {
        return Transform(Vec3(0.0f), glm::identity<Quat>(), Vec3(1.0f));
      }
    private:
      std::array<F32, 16> m_data;
    };
    inline Transform operator*(const Transform& t1, const Transform& t2) {
      if ((t1.getType() == TransformType::eTRS) && (t2.getType() == TransformType::eTRS)) {
        TransformTRSData trs1; t1.getTRS(trs1);
        TransformTRSData trs2; t2.getTRS(trs2);
        if (t1.isUniformScaling()) {
          TransformTRSData trs;
          trs.position = trs1.rotation * (trs1.scaling * trs2.position) + trs1.position;
          trs.rotation = trs1.rotation * trs2.rotation;
          trs.scaling  = trs1.scaling * trs2.scaling;
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
    using ArrayTransform = Array<Transform>;

#if defined(__cplusplus)
  }
}
#endif
