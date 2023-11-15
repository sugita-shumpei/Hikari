#ifndef HK_MATH_VEC__H
#define HK_MATH_VEC__H
#include "../data_type.h"
#include "../platform.h"
#include "../math/common.h"
#define HK_MATH_VEC_EQUAL_EPS 1e-5f
// Vec2
#define HK_MATH_VEC_CLONE_2(TYPE,V) HK_TYPE_INITIALIZER(TYPE,V.x,V.y)
#define HK_MATH_VEC_CLONE_3(TYPE,V) HK_TYPE_INITIALIZER(TYPE,V.x,V.y,V.z)
#define HK_MATH_VEC_CLONE_4(TYPE,V) HK_TYPE_INITIALIZER(TYPE,V.x,V.y,V.z,V.w)
#define HK_MATH_VEC2_INITIALIZER(X,Y) x{X},y{Y}
#define HK_MATH_VEC2_ARITHMETRIC_ASSIGN(V1,OP, V2)   V1.x OP V2.x; V1.y OP V2.y
#define HK_MATH_VEC2_ARITHMETRIC_ASSIGN_S(V1,OP, S2) V1.x OP   S2; V1.y OP   S2
#define HK_MATH_VEC2_ARITHMETRIC_INIT(V1,OP,V2)    HK_TYPE_INITIALIZER(HKVec2,V1.x OP V2.x, V1.y OP V2.y)
#define HK_MATH_VEC2_ARITHMETRIC_INIT_S1(S1,OP,V2) HK_TYPE_INITIALIZER(HKVec2,S1 OP V2.x, S1 OP V2.y)
#define HK_MATH_VEC2_ARITHMETRIC_INIT_S2(V1,OP,S2) HK_TYPE_INITIALIZER(HKVec2,V1.x OP S2, V1.y OP S2)
#define HK_MATH_VEC2_ARITHMETRIC_ADD(V1,OP,V2) (V1.x OP V2.x) + (V1.y OP V2.y)
#define HK_MATH_VEC2_ARITHMETRIC_MUL(V1,OP,V2) (V1.x OP V2.x) * (V1.y OP V2.y)
#define HK_MATH_VEC2_ARITHMETRIC_BIT_AND(V1,OP,V2) (V1.x OP V2.x) && (V1.y OP V2.y)
#define HK_MATH_VEC2_ARITHMETRIC_BIT_OR (V1,OP,V2) (V1.x OP V2.x) || (V1.y OP V2.y)
#define HK_MATH_VEC2_PASS_ARGUMENT_TO_STACK(V) V.x, V.y
#define HK_MATH_VEC2_ARITHMETRIC_PASS_ARGUMENT_TO_STACK(V1,OP,V2) (V1.x OP V2.x), (V1.y OP V2.y) 
// Vec3
#define HK_MATH_VEC3_INITIALIZER(X,Y,Z) x{X},y{Y},z{Z}
#define HK_MATH_VEC3_ARITHMETRIC_ASSIGN(V1,OP, V2)   V1.x OP V2.x; V1.y OP V2.y; V1.z OP V2.z
#define HK_MATH_VEC3_ARITHMETRIC_ASSIGN_S(V1,OP, S2) V1.x OP   S2; V1.y OP   S2; V1.z OP   S2
#define HK_MATH_VEC3_ARITHMETRIC_INIT(V1,OP,V2)    HK_TYPE_INITIALIZER(HKVec3,V1.x OP V2.x, V1.y OP V2.y, V1.z OP V2.z)
#define HK_MATH_VEC3_ARITHMETRIC_INIT_S1(S1,OP,V2) HK_TYPE_INITIALIZER(HKVec3,S1 OP V2.x, S1 OP V2.y, S1 OP V2.z)
#define HK_MATH_VEC3_ARITHMETRIC_INIT_S2(V1,OP,S2) HK_TYPE_INITIALIZER(HKVec3,V1.x OP S2, V1.y OP S2, V1.z OP S2)
#define HK_MATH_VEC3_ARITHMETRIC_ADD(V1,OP,V2) (V1.x OP V2.x) + (V1.y OP V2.y) + (V1.z OP V2.z)
#define HK_MATH_VEC3_ARITHMETRIC_MUL(V1,OP,V2) (V1.x OP V2.x) * (V1.y OP V2.y) * (V1.z OP V2.z)
#define HK_MATH_VEC3_ARITHMETRIC_BIT_AND(V1,OP,V2) (V1.x OP V2.x) && (V1.y OP V2.y) && (V1.z OP V2.z)
#define HK_MATH_VEC3_ARITHMETRIC_BIT_OR (V1,OP,V2) (V1.x OP V2.x) || (V1.y OP V2.y) || (V1.z OP V2.z)
#define HK_MATH_VEC3_PASS_ARGUMENT_TO_STACK(V) V.x, V.y, V.z
#define HK_MATH_VEC3_ARITHMETRIC_PASS_ARGUMENT_TO_STACK(V1,OP,V2) (V1.x OP V2.x), (V1.y OP V2.y), (V1.z OP V2.z)
// Vec4
#define HK_MATH_VEC4_INITIALIZER(X,Y,Z,W) x{X},y{Y},z{Z},w{W}
#define HK_MATH_VEC4_ARITHMETRIC_ASSIGN(V1,OP, V2)   V1.x OP V2.x; V1.y OP V2.y; V1.z OP V2.z; V1.w OP V2.w
#define HK_MATH_VEC4_ARITHMETRIC_ASSIGN_S(V1,OP, S2) V1.x OP   S2; V1.y OP   S2; V1.z OP   S2; V1.w OP   S2
#define HK_MATH_VEC4_ARITHMETRIC_INIT(V1,OP,V2)    HK_TYPE_INITIALIZER(HKVec4,V1.x OP V2.x, V1.y OP V2.y, V1.z OP V2.z, V1.w OP V2.w)
#define HK_MATH_VEC4_ARITHMETRIC_INIT_S1(S1,OP,V2) HK_TYPE_INITIALIZER(HKVec4,S1 OP V2.x, S1 OP V2.y, S1 OP V2.z, S1 OP V2.w)
#define HK_MATH_VEC4_ARITHMETRIC_INIT_S2(V1,OP,S2) HK_TYPE_INITIALIZER(HKVec4,V1.x OP S2, V1.y OP S2, V1.z OP S2, V1.w OP S2)
#define HK_MATH_VEC4_ARITHMETRIC_ADD(V1,OP,V2) (V1.x OP V2.x) + (V1.y OP V2.y) + (V1.z OP V2.z) + (V1.w OP V2.w)
#define HK_MATH_VEC4_ARITHMETRIC_MUL(V1,OP,V2) (V1.x OP V2.x) * (V1.y OP V2.y) * (V1.z OP V2.z) * (V1.w OP V2.w)
#define HK_MATH_VEC4_ARITHMETRIC_BIT_AND(V1,OP,V2) (V1.x OP V2.x) && (V1.y OP V2.y) && (V1.z OP V2.z) && (V1.w OP V2.w)
#define HK_MATH_VEC4_ARITHMETRIC_BIT_OR (V1,OP,V2) (V1.x OP V2.x) || (V1.y OP V2.y) || (V1.z OP V2.z) || (V1.w OP V2.w)
#define HK_MATH_VEC4_PASS_ARGUMENT_TO_STACK(V) V.x, V.y, V.z, V.w
#define HK_MATH_VEC4_ARITHMETRIC_PASS_ARGUMENT_TO_STACK(V1,OP,V2) (V1.x  OP V2.x), (V1.y OP V2.y), (V1.z OP  V2.z), (V1.w OP V2.w)
// Vec2
typedef struct HKCVec2 { HKF32 x; HKF32 y; } HKCVec2;
#if defined(__cplusplus)
typedef struct HKVec2 {
    HK_CXX11_CONSTEXPR HKVec2() HK_CXX_NOEXCEPT:HK_MATH_VEC2_INITIALIZER(0.0f,0.0f){}
    HK_CXX11_CONSTEXPR HKVec2(HKF32 v_) HK_CXX_NOEXCEPT : HK_MATH_VEC2_INITIALIZER(v_,v_) {}
    HK_CXX11_CONSTEXPR HKVec2(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT: HK_MATH_VEC2_INITIALIZER(x_, y_) {}
    HK_CXX11_CONSTEXPR HKVec2(const HKVec2&   v_) HK_CXX_NOEXCEPT : HK_MATH_VEC2_INITIALIZER(v_.x, v_.y) {}
    HK_CXX11_CONSTEXPR HKVec2(const HKCVec2& v_)HK_CXX_NOEXCEPT : HK_MATH_VEC2_INITIALIZER(v_.x, v_.y) {}
    HK_CXX14_CONSTEXPR HKVec2& operator=(const HKVec2&    v_) HK_CXX_NOEXCEPT { if (this != &v_) { HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_CXX14_CONSTEXPR HKVec2& operator=(const HKCVec2& v_) HK_CXX_NOEXCEPT {  { HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator+=(const HKVec2& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator-=(const HKVec2& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator*=(const HKVec2& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator/=(const HKVec2& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator*=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec2& operator/=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_CXX11_CONSTEXPR HKVec2 operator+() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, +x, +y); }
    HK_CXX11_CONSTEXPR HKVec2 operator-() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, -x, -y); }
    HK_CXX11_CONSTEXPR operator HKCVec2()const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCVec2, +x, +y); }
    HK_CXX11_CONSTEXPR HKF32  dot(const HKVec2& v) const  HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  lengthSqr() const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKF32  length() const    HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  distanceSqr(const HKVec2& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKF32  disrance(const HKVec2& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKVec2 normalize() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec2 ones()     HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec2 zeros()    HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec2 unitX()    HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec2 unitY()    HK_CXX_NOEXCEPT;
    HKF32 x;
    HKF32 y;
} HKVec2;
#else
typedef struct HKCVec2 HKVec2;
#endif
// Vec3
typedef struct HKCVec3 { HKF32 x; HKF32 y; HKF32 z;  } HKCVec3;
#if defined(__cplusplus)
typedef struct HKVec3 {
    HK_CXX11_CONSTEXPR HKVec3() HK_CXX_NOEXCEPT:x{ 0.0f }, y{ 0.0f }, z{ 0.0f } {}
    HK_CXX11_CONSTEXPR HKVec3(HKF32 v_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(v_, v_, v_) {}
    HK_CXX11_CONSTEXPR HKVec3(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(x_, y_, 0.0f) {}
    HK_CXX11_CONSTEXPR HKVec3(HKF32 x_, HKF32 y_, HKF32 z_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(x_, y_, z_) {}
    HK_CXX11_CONSTEXPR HKVec3(const HKVec3& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(v_.x, v_.y, v_.z) {}
    HK_CXX11_CONSTEXPR HKVec3(const HKCVec3& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(v_.x, v_.y, v_.z) {}
    HK_CXX11_CONSTEXPR HKVec3(const HKVec2& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(v_.x, v_.y, 0.0f) {}
    HK_CXX11_CONSTEXPR HKVec3(const HKVec2& v_, HKF32 z_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(v_.x, v_.y, z_) {}
    HK_CXX11_CONSTEXPR HKVec3(HKF32 x_, const HKVec2& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC3_INITIALIZER(x_, v_.x, v_.y) {}
    HK_CXX14_CONSTEXPR HKVec3& operator=(const HKVec3& v_) HK_CXX_NOEXCEPT { if (this != &v_) { HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_CXX14_CONSTEXPR HKVec3& operator=(const HKCVec3& v_) HK_CXX_NOEXCEPT {  { HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator+=(const HKVec3& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator-=(const HKVec3& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator*=(const HKVec3& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator/=(const HKVec3& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator*=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec3& operator/=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_CXX11_CONSTEXPR HKVec3 operator+() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, +x, +y, +z); }
    HK_CXX11_CONSTEXPR HKVec3 operator-() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, -x, -y, -z); }
    HK_CXX11_CONSTEXPR operator HKCVec3()const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCVec3, +x, +y, +z); }
    HK_CXX11_CONSTEXPR HKF32  dot(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  lengthSqr() const HK_CXX_NOEXCEPT;
    HK_INLINE HKF32                     length() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  distanceSqr(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKF32  disrance(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKVec3 normalize() const HK_CXX_NOEXCEPT;
    HK_CXX11_CONSTEXPR HKVec3 cross(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec3 ones()  HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec3 zeros() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec3 unitX() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec3 unitY() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec3 unitZ() HK_CXX_NOEXCEPT;
    HKF32 x;
    HKF32 y;
    HKF32 z;
} HKVec3;
#else
typedef struct HKCVec3 HKVec3;
#endif
// Vec4
typedef struct HKCVec4 { HKF32 x; HKF32 y; HKF32 z; HKF32 w; } HKCVec4;
#if defined(__cplusplus)
typedef struct HKVec4 {
    HK_CXX11_CONSTEXPR HKVec4() HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(0.0f,0.0f,0.0f,0.0f){}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 v_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(v_, v_, v_,v_) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_,y_,0.0f,0.0f) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, HKF32 y_, HKF32 z_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_, y_, z_, 0.0f) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, HKF32 y_, HKF32 z_, HKF32 w_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_, y_, z_, w_) {}
    HK_CXX11_CONSTEXPR HKVec4(const HKVec4& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(v_.x, v_.y, v_.z, v_.w) {}
    HK_CXX11_CONSTEXPR HKVec4(const HKCVec4& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(v_.x, v_.y, v_.z, v_.w) {}
    HK_CXX11_CONSTEXPR HKVec4(const HKVec2& v_, HKF32 z_, HKF32 w_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(v_.x, v_.y,z_,w_) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, const HKVec2& v_, HKF32 w_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_,v_.x, v_.y,w_) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, HKF32 y_, const HKVec2& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_,y_,v_.x, v_.y) {}
    HK_CXX11_CONSTEXPR HKVec4(const HKVec2& xy_, const HKVec2& zw_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(xy_.x, xy_.y,zw_.x,zw_.y) {}
    HK_CXX11_CONSTEXPR HKVec4(const HKVec3& v_, HKF32 w_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(v_.x, v_.y, v_.z,w_) {}
    HK_CXX11_CONSTEXPR HKVec4(HKF32 x_, const HKVec3& v_) HK_CXX_NOEXCEPT : HK_MATH_VEC4_INITIALIZER(x_,v_.x, v_.y, v_.z) {}
    HK_CXX14_CONSTEXPR HKVec4& operator=(const HKVec4& v_) HK_CXX_NOEXCEPT { if (this != &v_) { HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_CXX14_CONSTEXPR HKVec4& operator=(const HKCVec4& v_) HK_CXX_NOEXCEPT {  { HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*this), =, v_); } return *this; }
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator+=(const HKVec4& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator-=(const HKVec4& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator*=(const HKVec4& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator/=(const HKVec4& v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator*=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKVec4& operator/=(HKF32 v) HK_CXX_NOEXCEPT;
    HK_CXX11_CONSTEXPR HKVec4 operator+() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, +x, +y, +z, +w); }
    HK_CXX11_CONSTEXPR HKVec4 operator-() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, -x, -y, -z, -w); }
    HK_CXX11_CONSTEXPR operator HKCVec4() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCVec4, +x, +y, +z, +w); }
    HK_CXX11_CONSTEXPR HKF32  dot(const HKVec4& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  lengthSqr() const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKF32  length() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKF32  distanceSqr(const HKVec4& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKF32  disrance(const HKVec4& v) const HK_CXX_NOEXCEPT;
    HK_INLINE                    HKVec4 normalize() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 ones()  HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 zeros() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 unitX() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 unitY() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 unitZ() HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR static HKVec4 unitW() HK_CXX_NOEXCEPT;
    HKF32 x;
    HKF32 y;
    HKF32 z;
    HKF32 w;
} HKVec4;
#else
typedef struct HKCVec4 HKVec4;
#endif
// Vec2
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_create()HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, 0.0f, 0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_create1(HKF32 v_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, v_, v_); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_create2(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, x_, y_); }
//HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec2_equal(const HKVec2 * v1, const HKVec2 * v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_BIT_AND((*v1), == , (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assign(HKVec2* v1, const HKVec2* v2)HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*v1), =, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignAdd(HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*v1), +=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignSub(HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*v1), -=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignMul(HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*v1), *=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignDiv(HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN((*v1), /=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignMul_s(HKVec2* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN_S((*v1), *=, s2); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKVec2_assignDiv_s(HKVec2* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC2_ARITHMETRIC_ASSIGN_S((*v1), /=, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_add(const HKVec2* v1, const HKVec2* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT((*v1), +, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_sub(const HKVec2* v1, const HKVec2* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT((*v1), -, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_mul(const HKVec2* v1, const HKVec2* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT((*v1), *, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_div(const HKVec2* v1, const HKVec2* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT((*v1), /, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_mul_s1(HKF32 s1, const HKVec2* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT_S1(s1, *, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_mul_s2(const HKVec2* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT_S2((*v1), *, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_div_s2(const HKVec2* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_INIT_S2((*v1), /, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_dot(const HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_ADD((*v1), *, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_dot_stack_1(const HKVec2 HK_REF v1, const HKVec2* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_ADD((v1), *, (*v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_dot_stack_2(const HKVec2* v1, const HKVec2 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_ADD((*v1), *, (v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_dot_stack(const HKVec2 HK_REF v1, const HKVec2 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_ADD((v1), *, (v2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_lengthSqr(const HKVec2* v) HK_CXX_NOEXCEPT { return HK_MATH_VEC2_ARITHMETRIC_ADD((*v), *,(*v)); }
HK_INLINE                    HKF32  HKVec2_length(const HKVec2* v) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec2_lengthSqr(v)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2_distanceSqr(const HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { return HK_POW2(v1->x - v2->x) + HK_POW2(v1->y - v2->y); }
HK_INLINE                    HKF32  HKVec2_distance(const HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec2_distanceSqr(v1,v2)); }
HK_INLINE                    HKVec2 HKVec2_normalize(const HKVec2* v) HK_CXX_NOEXCEPT { return HKVec2_mul_s2(v,1.0f/ HKVec2_length(v)); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec2_equal_withEps(const HKVec2* v1, const HKVec2* v2, HKF32 eps) HK_CXX_NOEXCEPT {
    return HKVec2_distanceSqr(v1, v2) < HKMath_fmaxf(HKVec2_lengthSqr(v1), HKVec2_lengthSqr(v2)) * eps * eps;
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec2_equal(const HKVec2* v1, const HKVec2* v2) HK_CXX_NOEXCEPT {
    return HKVec2_equal_withEps(v1, v2, HK_MATH_VEC_EQUAL_EPS);
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_zeros() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, 0.0f, 0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_ones()  HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, 1.0f, 1.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_unitX() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, 1.0f, 0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2_unitY() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec2, 0.0f, 1.0f); }
 // Vec3
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_cross(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, (*v1).y * (*v2).z - (*v1).z * (*v2).y, (*v1).z * (*v2).x - (*v1).x * (*v2).z, (*v1).x * (*v2).y - (*v1).y * (*v2).x); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_cross_stack_1(HKVec3 v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, v1.y * (*v2).z - v1.z * (*v2).y, v1.z * (*v2).x - v1.x * (*v2).z, v1.x * (*v2).y - v1.y * (*v2).x); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_cross_stack_2(const HKVec3* v1, HKVec3 v2) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, (*v1).y * v2.z - (*v1).z * v2.y, (*v1).z * v2.x - (*v1).x * v2.z, (*v1).x * v2.y - (*v1).y * v2.x); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_create()HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 0.0f, 0.0f,0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_create1(HKF32 v_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, v_, v_, v_); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_create2(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, x_, y_,0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_create3(HKF32 x_, HKF32 y_, HKF32 z_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, x_, y_, z_); }
//HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec3_equal(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_BIT_AND((*v1), == , (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assign(HKVec3* v1, const HKVec3* v2)HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*v1), =, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignAdd(HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*v1), +=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignSub(HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*v1), -=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignMul(HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*v1), *=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignDiv(HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN((*v1), /=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignMul_s(HKVec3* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN_S((*v1), *=, s2); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec3_assignDiv_s(HKVec3* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC3_ARITHMETRIC_ASSIGN_S((*v1), /=, s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_add(const HKVec3* v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), +, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_sub(const HKVec3* v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), -, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_mul(const HKVec3* v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_div(const HKVec3* v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), /, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_add_stack_1(const HKVec3 HK_REF v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((v1), +, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_sub_stack_1(const HKVec3 HK_REF v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((v1), -, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_mul_stack_1(const HKVec3 HK_REF v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_div_stack_1(const HKVec3 HK_REF v1, const HKVec3* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((v1), / , (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_add_stack_2(const HKVec3* v1, const HKVec3 HK_REF v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), +, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_sub_stack_2(const HKVec3* v1, const HKVec3 HK_REF v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), -, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_mul_stack_2(const HKVec3* v1, const HKVec3 HK_REF v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), *, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_div_stack_2(const HKVec3* v1, const HKVec3 HK_REF v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT((*v1), / ,(v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_mul_s1(HKF32 s1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT_S1(s1, *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_mul_s2(const HKVec3* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT_S2((*v1), *, s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_div_s2(const HKVec3* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_INIT_S2((*v1), / , s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_dot(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_ADD((*v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_dot_stack_1(const HKVec3 HK_REF v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_ADD((v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_dot_stack_2(const HKVec3* v1, const HKVec3 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_ADD((*v1), *, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_dot_stack(const HKVec3 HK_REF v1, const HKVec3 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_ADD((v1), *, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_lengthSqr(const HKVec3* v) HK_CXX_NOEXCEPT { return HK_MATH_VEC3_ARITHMETRIC_ADD((*v), *, (*v)); }
 HK_INLINE                    HKF32  HKVec3_length(const HKVec3* v) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec3_lengthSqr(v)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec3_distanceSqr(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HK_POW2(v1->x - v2->x) + HK_POW2(v1->y - v2->y); }
 HK_INLINE                    HKF32  HKVec3_distance(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec3_distanceSqr(v1, v2)); }
 HK_INLINE                    HKVec3 HKVec3_normalize(const HKVec3* v) HK_CXX_NOEXCEPT { return HKVec3_mul_s2(v, 1.0f / HKVec3_length(v)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec3_equal_withEps(const HKVec3* v1, const HKVec3* v2, HKF32 eps) HK_CXX_NOEXCEPT {
     return HKVec3_distanceSqr(v1, v2) < HKMath_fmaxf(HKVec3_lengthSqr(v1), HKVec3_lengthSqr(v2)) * eps * eps;
 }
 HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec3_equal(const HKVec3* v1, const HKVec3* v2) HK_CXX_NOEXCEPT {
     return HKVec3_equal_withEps(v1, v2, HK_MATH_VEC_EQUAL_EPS);
 }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_zeros() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 0.0f, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_ones () HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 1.0f, 1.0f, 1.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_unitX() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 1.0f, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_unitY() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 0.0f, 1.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3_unitZ() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, 0.0f, 0.0f, 1.0f); }
 // Vec4
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_create()HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 0.0f, 0.0f,0.0f,0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_create1(HKF32 v_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, v_, v_, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_create2(HKF32 x_, HKF32 y_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, x_, y_, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_create3(HKF32 x_, HKF32 y_, HKF32 z_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, x_, y_, z_, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_create4(HKF32 x_, HKF32 y_, HKF32 z_, HKF32 w_) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, x_, y_, z_, w_); }
// HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec4_equal(const HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_BIT_AND((*v1), == , (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assign(HKVec4* v1, const HKVec4* v2)HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), =, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignAdd(HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), +=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignSub(HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), -=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignMul(HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), *=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignDiv(HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), /=, (*v2)); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignMul_s(HKVec4* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN_S((*v1), *=, s2); }
 HK_INLINE HK_CXX14_CONSTEXPR void   HKVec4_assignDiv_s(HKVec4* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN_S((*v1), /=, s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_add(const HKVec4* v1, const HKVec4* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT((*v1), +, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_sub(const HKVec4* v1, const HKVec4* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT((*v1), -, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_mul(const HKVec4* v1, const HKVec4* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT((*v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_div(const HKVec4* v1, const HKVec4* v2)HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT((*v1), / , (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_mul_s1(HKF32 s1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT_S1(s1, *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_mul_s2(const HKVec4* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT_S2((*v1), *, s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_div_s2(const HKVec4* v1, HKF32 s2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_INIT_S2((*v1), / , s2); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_dot(const HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_ADD((*v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_dot_stack_1(const HKVec4 HK_REF v1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_ADD((v1), *, (*v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_dot_stack_2(const HKVec4* v1, const HKVec4 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_ADD((*v1), *, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_dot_stack(const HKVec4 HK_REF v1, const HKVec4 HK_REF v2) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_ADD((v1), *, (v2)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_lengthSqr(const HKVec4* v) HK_CXX_NOEXCEPT { return HK_MATH_VEC4_ARITHMETRIC_ADD((*v), *, (*v)); }
 HK_INLINE                    HKF32  HKVec4_length(const HKVec4* v)    HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec4_lengthSqr(v)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4_distanceSqr(const HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HK_POW2(v1->x - v2->x) + HK_POW2(v1->y - v2->y); }
 HK_INLINE                    HKF32  HKVec4_distance(const HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKVec4_distanceSqr(v1, v2)); }
 HK_INLINE                    HKVec4 HKVec4_normalize(const HKVec4* v) HK_CXX_NOEXCEPT { return HKVec4_mul_s2(v, 1.0f / HKVec4_length(v)); }
 HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec4_equal_withEps(const HKVec4* v1, const HKVec4* v2, HKF32 eps) HK_CXX_NOEXCEPT {
     return HKVec4_distanceSqr(v1, v2) < HKMath_fmaxf(HKVec4_lengthSqr(v1), HKVec4_lengthSqr(v2)) * eps * eps;
 }
 HK_INLINE HK_CXX11_CONSTEXPR HKBool HKVec4_equal(const HKVec4* v1, const HKVec4* v2) HK_CXX_NOEXCEPT {
     return HKVec4_equal_withEps(v1, v2, HK_MATH_VEC_EQUAL_EPS);
 }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_zeros() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 0.0f, 0.0f, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_ones()  HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 1.0f, 1.0f, 1.0f, 1.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_unitX() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 1.0f, 0.0f, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_unitY() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 0.0f, 1.0f, 0.0f, 0.0f); }
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_unitZ() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 0.0f, 0.0f, 1.0f, 0.0f); } 
 HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4_unitW() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec4, 0.0f, 0.0f, 0.0f, 1.0f); }
#if defined(__cplusplus)
// Vec2
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator==(const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return  HKVec2_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator!=(const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return !HKVec2_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator+ (const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return HKVec2_add(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator- (const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return HKVec2_sub(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator* (const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return HKVec2_mul(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator/ (const HKVec2& v1, const HKVec2& v2) HK_CXX_NOEXCEPT { return HKVec2_div(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator* (HKF32         s1, const HKVec2& v2) HK_CXX_NOEXCEPT { return HKVec2_mul_s1(s1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator* (const HKVec2& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec2_mul_s2(&v1, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 operator/ (const HKVec2& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec2_div_s2(&v1, s2); }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator+=(const HKVec2& v) HK_CXX_NOEXCEPT { HKVec2_assignAdd(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator-=(const HKVec2& v) HK_CXX_NOEXCEPT { HKVec2_assignSub(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator*=(const HKVec2& v) HK_CXX_NOEXCEPT { HKVec2_assignMul(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator/=(const HKVec2& v) HK_CXX_NOEXCEPT { HKVec2_assignDiv(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator*=(HKF32 v) HK_CXX_NOEXCEPT { HKVec2_assignMul_s(this, v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec2& HKVec2::operator/=(HKF32 v) HK_CXX_NOEXCEPT { HKVec2_assignDiv_s(this, v); return *this; }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2::dot(const HKVec2& v) const HK_CXX_NOEXCEPT { return HKVec2_dot(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec2::lengthSqr() const HK_CXX_NOEXCEPT { return HKVec2_lengthSqr(this); }
HK_INLINE                    HKF32  HKVec2::length() const    HK_CXX_NOEXCEPT { return HKVec2_length(this); }
HK_INLINE                    HKVec2 HKVec2::normalize() const HK_CXX_NOEXCEPT { return HKVec2_normalize(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2::ones()  HK_CXX_NOEXCEPT { return HKVec2_ones();  }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2::zeros() HK_CXX_NOEXCEPT { return HKVec2_zeros(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2::unitX() HK_CXX_NOEXCEPT { return HKVec2_unitX(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec2 HKVec2::unitY() HK_CXX_NOEXCEPT { return HKVec2_unitY(); }
// Vec3
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator==(const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return  HKVec3_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator!=(const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return !HKVec3_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator+ (const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKVec3_add(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator- (const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKVec3_sub(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator* (const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKVec3_mul(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator/ (const HKVec3& v1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKVec3_div(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator* (HKF32         s1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKVec3_mul_s1(s1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator* (const HKVec3& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec3_mul_s2(&v1, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 operator/ (const HKVec3& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec3_div_s2(&v1, s2); }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator+=(const HKVec3& v) HK_CXX_NOEXCEPT { HKVec3_assignAdd(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator-=(const HKVec3& v) HK_CXX_NOEXCEPT { HKVec3_assignSub(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator*=(const HKVec3& v) HK_CXX_NOEXCEPT { HKVec3_assignMul(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator/=(const HKVec3& v) HK_CXX_NOEXCEPT { HKVec3_assignDiv(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator*=(HKF32 v) HK_CXX_NOEXCEPT { HKVec3_assignMul_s(this, v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec3& HKVec3::operator/=(HKF32 v) HK_CXX_NOEXCEPT { HKVec3_assignDiv_s(this, v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKF32  HKVec3::dot(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKVec3_dot(this, &v); }
HK_INLINE HK_CXX14_CONSTEXPR HKF32  HKVec3::lengthSqr() const HK_CXX_NOEXCEPT { return HKVec3_lengthSqr(this); }
HK_INLINE                    HKF32  HKVec3::length() const HK_CXX_NOEXCEPT { return  HKVec3_length(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::cross(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKVec3_cross(this, &v); }
HK_INLINE                    HKVec3 HKVec3::normalize() const HK_CXX_NOEXCEPT { return HKVec3_normalize(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::ones()  HK_CXX_NOEXCEPT { return HKVec3_ones(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::zeros() HK_CXX_NOEXCEPT { return HKVec3_zeros(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::unitX() HK_CXX_NOEXCEPT { return HKVec3_unitX(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::unitY() HK_CXX_NOEXCEPT { return HKVec3_unitY(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKVec3::unitZ() HK_CXX_NOEXCEPT { return HKVec3_unitZ(); }
// Vec4
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator==(const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return  HKVec4_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator!=(const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return !HKVec4_equal(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator+ (const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return HKVec4_add(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator- (const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return HKVec4_sub(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator* (const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return HKVec4_mul(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator/ (const HKVec4& v1, const HKVec4& v2) HK_CXX_NOEXCEPT { return HKVec4_div(&v1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator* (HKF32         s1, const HKVec4& v2) HK_CXX_NOEXCEPT { return HKVec4_mul_s1(s1, &v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator* (const HKVec4& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec4_mul_s2(&v1, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 operator/ (const HKVec4& v1, HKF32         s2) HK_CXX_NOEXCEPT { return HKVec4_div_s2(&v1, s2); }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator+=(const HKVec4& v) HK_CXX_NOEXCEPT { HKVec4_assignAdd(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator-=(const HKVec4& v) HK_CXX_NOEXCEPT { HKVec4_assignSub(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator*=(const HKVec4& v) HK_CXX_NOEXCEPT { HKVec4_assignMul(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator/=(const HKVec4& v) HK_CXX_NOEXCEPT { HKVec4_assignDiv(this, &v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator*=(HKF32 v) HK_CXX_NOEXCEPT { HKVec4_assignMul_s(this, v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKVec4& HKVec4::operator/=(HKF32 v) HK_CXX_NOEXCEPT { HKVec4_assignDiv_s(this, v); return *this; }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4::dot(const HKVec4& v) const HK_CXX_NOEXCEPT { return HKVec4_dot(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKVec4::lengthSqr() const HK_CXX_NOEXCEPT { return HKVec4_lengthSqr(this); }
HK_INLINE                    HKF32  HKVec4::length()    const HK_CXX_NOEXCEPT { return HKVec4_length(this); }
HK_INLINE                    HKVec4 HKVec4::normalize() const HK_CXX_NOEXCEPT { return HKVec4_normalize(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::ones()  HK_CXX_NOEXCEPT { return HKVec4_ones(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::zeros() HK_CXX_NOEXCEPT { return HKVec4_zeros(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::unitX() HK_CXX_NOEXCEPT { return HKVec4_unitX(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::unitY() HK_CXX_NOEXCEPT { return HKVec4_unitY(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::unitZ() HK_CXX_NOEXCEPT { return HKVec4_unitZ(); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4 HKVec4::unitW() HK_CXX_NOEXCEPT { return HKVec4_unitW(); }
#endif

#endif
