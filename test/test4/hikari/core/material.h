#pragma once
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    struct MaterialObject : public Object {
      virtual ~MaterialObject() noexcept {}
      void setValue(const Str& name, Bool v);
      void setValue(const Str& name, F32 f32);
      void setValue(const Str& name, I32 i32);
      void setValue(const Str& name, const Vec2& v);
      void setValue(const Str& name, const Vec3& v);
      void setValue(const Str& name, const Vec4& v);
      void setValue(const Str& name, const Mat2& v);
      void setValue(const Str& name, const Mat3& v);
      void setValue(const Str& name, const Mat4& v);
      void setValue(const Str& name, const Array<F32>& array_f32);
      void setValue(const Str& name, const Array<I32>& array_i32);
      void setValue(const Str& name, const Array<Vec2>& v);
      void setValue(const Str& name, const Array<Vec3>& v);
      void setValue(const Str& name, const Array<Vec4>& v);
      void setValue(const Str& name, const Array<Mat2>& v);
      void setValue(const Str& name, const Array<Mat3>& v);
      void setValue(const Str& name, const Array<Mat4>& v);
    };
  }
}
