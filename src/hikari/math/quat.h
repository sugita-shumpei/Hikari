#ifndef HK_MATH_QUAT__H
#define HK_MATH_QUAT__H
#include "vec.h"
#include "matrix.h"
#define HK_MATH_QUAT_EQUAL_EPS 1e-3f
#define HK_MATH_QUAT_ARITHMETRIC_INIT(V1,OP,V2)    HK_TYPE_INITIALIZER(HKQuat,V1.w OP V2.w, V1.x OP V2.x, V1.y OP V2.y, V1.z OP V2.z)
#define HK_MATH_QUAT_ARITHMETRIC_INIT_S1(S1,OP,V2) HK_TYPE_INITIALIZER(HKQuat,S1 OP V2.w, S1 OP V2.x, S1 OP V2.y, S1 OP V2.z)
#define HK_MATH_QUAT_ARITHMETRIC_INIT_S2(V1,OP,S2) HK_TYPE_INITIALIZER(HKQuat,V1.w OP S2, V1.x OP S2, V1.y OP S2, V1.z OP S2)

typedef struct HKCQuat { HKF32 w; HKF32 x; HKF32 y; HKF32 z; } HKCQuat;
#if defined(__cplusplus)
typedef struct HKQuat {
	HK_CXX11_CONSTEXPR HKQuat() HK_CXX_NOEXCEPT;
	HK_CXX11_CONSTEXPR HKQuat(HKF32 w_) HK_CXX_NOEXCEPT:x{ 0.0f }, y{ 0.0f }, z{ 0.0f }, w{ w_ } {}
	HK_CXX11_CONSTEXPR HKQuat(HKF32 w_, HKF32 x_, HKF32 y_, HKF32 z_) HK_CXX_NOEXCEPT : x{ x_ }, y{ y_ }, z{ z_ }, w{ w_ } {}
	HK_CXX11_CONSTEXPR HKQuat(const HKQuat& q) HK_CXX_NOEXCEPT : x{ q.x }, y{ q.y }, z{ q.z }, w{ q.w } {}
	HK_CXX11_CONSTEXPR HKQuat(const HKCQuat& q) HK_CXX_NOEXCEPT : x{ q.x }, y{ q.y }, z{ q.z }, w{ q.w } {}
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator+() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,+w, +x, +y, +z); }
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator-() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,-w, -x, -y, -z); }
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator=(const HKQuat& q)  HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator=(const HKCQuat& q)  HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator+=(const HKQuat& q) HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator-=(const HKQuat& q) HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator*=(const HKQuat& q) HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator*=(HKF32 v) HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX14_CONSTEXPR HKQuat& operator/=(HKF32 v) HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR operator HKCQuat() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCQuat,+w, +x, +y, +z); }
	HK_INLINE HK_CXX11_CONSTEXPR HKVec3 rotateVector(const HKVec3& v) const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKF32  lengthSqr()const HK_CXX_NOEXCEPT;
	HK_INLINE                    HKF32  length()const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKF32  distanceSqr(const HKQuat& q)const HK_CXX_NOEXCEPT;
	HK_INLINE                    HKF32  distance(const HKQuat& q)const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat inverse  ()const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat conjugate()const HK_CXX_NOEXCEPT;
	HK_INLINE                    HKQuat normalize()const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat cross(const HKQuat& q)const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKQuat dot(const HKQuat& q)const HK_CXX_NOEXCEPT;
	HK_INLINE                    HKVec3 eulerAngles() const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKMat3x3 toMat3x3() const HK_CXX_NOEXCEPT;
	HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 toMat4x4() const HK_CXX_NOEXCEPT;
	static HK_INLINE HK_CXX11_CONSTEXPR HKQuat identity() HK_CXX_NOEXCEPT;
	static HK_INLINE HK_CXX11_CONSTEXPR HKQuat zeros() HK_CXX_NOEXCEPT;
	static HK_INLINE                    HKQuat euler(HKF32 rx, HKF32 ry, HKF32 rz) HK_CXX_NOEXCEPT;
	static HK_INLINE                    HKQuat mat3x3(const HKMat3x3& m) HK_CXX_NOEXCEPT;
	static HK_INLINE                    HKQuat mat4x4(const HKMat4x4& m) HK_CXX_NOEXCEPT;
	HKF32 w;
	HKF32 x;
	HKF32 y;
	HKF32 z;
} HKQuat; 
#else
typedef struct HKCQuat HKQuat;
#endif

HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_create() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,0.0f,0.0f,0.0f,0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_create1(HKF32 w) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat, w, 0.0f, 0.0f, 0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_create4(HKF32 w, HKF32 x, HKF32 y, HKF32 z) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat, w, x, y, z); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_add(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT { return HK_MATH_QUAT_ARITHMETRIC_INIT((*q1),+,(*q2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_sub(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT { return HK_MATH_QUAT_ARITHMETRIC_INIT((*q1),-,(*q2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_mul(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT {
	return HK_TYPE_INITIALIZER(HKQuat,
		(q1->w * q2->w - q1->x * q2->x - q1->y * q2->y - q1->z * q2->z),
		((q1->w * q2->x + q2->w * q1->x) + (q1->y * q2->z - q1->z * q2->y)),
		((q1->w * q2->y + q2->w * q1->y) + (q1->z * q2->x - q1->x * q2->z)),
		((q1->w * q2->z + q2->w * q1->z) + (q1->x * q2->y - q1->y * q2->x)));
}
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_mul_stack_2(const HKQuat* q1, HKQuat q2) HK_CXX_NOEXCEPT {
	return HK_TYPE_INITIALIZER(HKQuat,
		(q1->w * q2.w - q1->x * q2.x - q1->y * q2.y - q1->z * q2.z),
		((q1->w * q2.x + q2.w * q1->x) + (q1->y * q2.z - q1->z * q2.y)),
		((q1->w * q2.y + q2.w * q1->y) + (q1->z * q2.x - q1->x * q2.z)),
		((q1->w * q2.z + q2.w * q1->z) + (q1->x * q2.y - q1->y * q2.x)));
}
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_mul_s1(HKF32 v, const HKQuat* q) HK_CXX_NOEXCEPT { return HK_MATH_QUAT_ARITHMETRIC_INIT_S1(v,*,(*q)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_mul_s2(const HKQuat* q, HKF32 v) HK_CXX_NOEXCEPT { return HK_MATH_QUAT_ARITHMETRIC_INIT_S2((*q),*,v); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_div_s2(const HKQuat* q, HKF32 v) HK_CXX_NOEXCEPT { return HK_MATH_QUAT_ARITHMETRIC_INIT_S2((*q),/,v); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_mul_stack_s2(HKF32 qw, HKF32 qx, HKF32 qy, HKF32 qz, HKF32 v) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,v*qw,v*qx,v*qy,v*qz); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignAdd(HKQuat* v1, const HKQuat* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), +=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assign(HKQuat* v1, const HKQuat* v2)HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), =, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignSub(HKQuat* v1, const HKQuat* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), -=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignMul(HKQuat* v1, const HKQuat* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } *v1 = HKQuat_mul(v1,v2); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignDiv(HKQuat* v1, const HKQuat* v2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN((*v1), /=, (*v2)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignMul_s(HKQuat* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN_S((*v1), *=, s2); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKQuat_assignDiv_s(HKQuat* v1, HKF32 s2) HK_CXX_NOEXCEPT { if (!v1) { return; } HK_MATH_VEC4_ARITHMETRIC_ASSIGN_S((*v1), /=, s2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKQuat_rotateVector(const HKQuat* q1, const HKVec3* v2) HK_CXX_NOEXCEPT { return HKMat3x3_mulVector_v2_stack_m1(HK_MATH_MAT3X3_TYPE_INITIALIZER(
	2.0f *HK_POW2(q1->w) + 2.0f * HK_POW2(q1->x) - 1.0f, 2.0f *(q1->x*q1->y+q1->z*q1->w), 2.0f* (q1->x * q1->z - q1->y * q1->w),
	2.0f *(q1->x*q1->y-q1->z*q1->w), 2.0f * HK_POW2(q1->w) + 2.0f * HK_POW2(q1->y) - 1.0f,2.0f* (q1->y * q1->z + q1->x * q1->w),
	2.0f* (q1->x * q1->z + q1->y * q1->w), 2.0f * (q1->y * q1->z - q1->x * q1->w), 2.0f * HK_POW2(q1->w) + 2.0f * HK_POW2(q1->z) - 1.0f
),v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKQuat_rotateVector_stack(const HKQuat* q1, const HKVec3 HK_REF v2) HK_CXX_NOEXCEPT {
	return HKMat3x3_mulVector_v2_stack(HK_MATH_MAT3X3_TYPE_INITIALIZER(
		2.0f * HK_POW2(q1->w) + 2.0f * HK_POW2(q1->x) - 1.0f, 2.0f * (q1->x * q1->y + q1->z * q1->w), 2.0f * (q1->x * q1->z - q1->y * q1->w),
		2.0f * (q1->x * q1->y - q1->z * q1->w), 2.0f * HK_POW2(q1->w) + 2.0f * HK_POW2(q1->y) - 1.0f, 2.0f * (q1->y * q1->z + q1->x * q1->w),
		2.0f * (q1->x * q1->z + q1->y * q1->w), 2.0f * (q1->y * q1->z - q1->x * q1->w), 2.0f * HK_POW2(q1->w) + 2.0f * HK_POW2(q1->z) - 1.0f
	), v2);
}
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKQuat_lengthSqr(const HKQuat* q) HK_CXX_NOEXCEPT { return HK_POW2(q->x) + HK_POW2(q->y) + HK_POW2(q->z) + HK_POW2(q->w); }
HK_INLINE                    HKF32  HKQuat_length(const HKQuat* q) HK_CXX_NOEXCEPT  { return HKMath_sqrtf(HKQuat_lengthSqr(q)); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKQuat_distanceSqr(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT { return HK_POW2(q1->x-q2->x) + HK_POW2(q1->y-q2->y) + HK_POW2(q1->z-q2->z) + HK_POW2(q1->w-q2->w); }
HK_INLINE                    HKF32  HKQuat_distance(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT { return HKMath_sqrtf(HKQuat_distanceSqr(q1,q2)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_inverse(const HKQuat* q) HK_CXX_NOEXCEPT { return HKQuat_mul_stack_s2(q->w, -q->x, -q->y, -q->z, 1.0f / HKQuat_lengthSqr(q)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_div(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT {
	return HKQuat_mul_stack_2(q1, HKQuat_inverse(q2));
}
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_conjugate(const HKQuat* q) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,q->w,-q->x,-q->y,-q->z);}
HK_INLINE                    HKQuat HKQuat_normalize(const HKQuat* q) HK_CXX_NOEXCEPT { return HKQuat_mul_stack_s2(q->w, q->x, q->y, q->z, 1.0f / HKQuat_length(q)); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_cross(const HKQuat* q1, const HKQuat* q2) HK_CXX_NOEXCEPT { return HKQuat_mul(q1, q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKQuat_dot(const HKQuat* q1, const HKQuat* q2)HK_CXX_NOEXCEPT { return (q1->w * q2->w - q1->x * q2->x - q1->y * q2->y - q1->z * q2->z); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat3x3 HKQuat_toMat3x3(const HKQuat* q) HK_CXX_NOEXCEPT {
	return HK_MATH_MAT3X3_TYPE_INITIALIZER(2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->x) - 1.0f, 2.0f * (q->x * q->y + q->z * q->w), 2.0f * (q->x * q->z - q->y * q->w),
		2.0f * (q->x * q->y - q->z * q->w), 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->y) - 1.0f, 2.0f * (q->y * q->z + q->x * q->w),
		2.0f * (q->x * q->z + q->y * q->w), 2.0f * (q->y * q->z - q->x * q->w), 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->z) - 1.0f);
}
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 HKQuat_toMat4x4(const HKQuat* q) HK_CXX_NOEXCEPT {
	return HK_MATH_MAT4X4_TYPE_INITIALIZER(2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->x) - 1.0f, 2.0f * (q->x * q->y + q->z * q->w), 2.0f * (q->x * q->z - q->y * q->w),0.0f,
		2.0f * (q->x * q->y - q->z * q->w), 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->y) - 1.0f, 2.0f * (q->y * q->z + q->x * q->w), 0.0f,
		2.0f * (q->x * q->z + q->y * q->w), 2.0f * (q->y * q->z - q->x * q->w), 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->z) - 1.0f, 0.0f, 
		0.0f, 0.0f, 0.0f,1.0f );
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKQuat_equal_withEps(const HKQuat* q1, const HKQuat* q2, HKF32 eps) HK_CXX_NOEXCEPT {
	return HKQuat_distanceSqr(q1, q2) < HKMath_fmaxf(HKQuat_lengthSqr(q1), HKQuat_lengthSqr(q2)) * eps * eps;
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKQuat_equal(const HKQuat* v1, const HKQuat* v2) HK_CXX_NOEXCEPT {
	return HKQuat_equal_withEps(v1, v2, HK_MATH_QUAT_EQUAL_EPS);
}
HK_INLINE                    HKVec3 HKQuat_eulerAngles(const HKQuat* q) HK_CXX_NOEXCEPT {
	HKF32 len2  = HKQuat_lengthSqr(q);
	HKF32 sin_x = (2.0f * q->y * q->z + 2.0f * q->x * q->w) / len2;
	HKF32 cos_x2 = 1.0f - sin_x * sin_x;
	HKF32 rx    = HK_M_RAD_2_DEG * HKMath_asinf(sin_x);
	HKF32 ry    = 0.0f;
	HKF32 rz    = 0.0f;
	if (cos_x2 > 0.0f) {
		ry = HK_M_RAD_2_DEG * HKMath_atan2f(-2.0f * q->x * q->z + 2.0f * q->y * q->w, 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->z) - len2);
		rz = HK_M_RAD_2_DEG * HKMath_atan2f(-2.0f * q->x * q->y + 2.0f * q->z * q->w, 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->y) - len2);
	}
	else {
		rz = HK_M_RAD_2_DEG * HKMath_atan2f(2.0f * q->x * q->y + 2.0f * q->z * q->w, 2.0f * HK_POW2(q->w) + 2.0f * HK_POW2(q->x) - len2);
	}
	return HK_TYPE_INITIALIZER(HKVec3, rx, ry, rz);
}
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_identity() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,1.0f,0.0f,0.0f,0.0f); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat_zeros() HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKQuat,0.0f,0.0f,0.0f,0.0f); }
HK_INLINE                    HKQuat HKQuat_euler(HKF32 rx, HKF32 ry, HKF32 rz) HK_CXX_NOEXCEPT {
	HKF32 cx = HKMath_cosf(((HKF32)HK_M_DEG_2_RAD) * rx * 0.5f); HKF32 sx = HKMath_sinf(((HKF32)HK_M_DEG_2_RAD) * rx * 0.5f);
	HKF32 cy = HKMath_cosf(((HKF32)HK_M_DEG_2_RAD) * ry * 0.5f); HKF32 sy = HKMath_sinf(((HKF32)HK_M_DEG_2_RAD) * ry * 0.5f);
	HKF32 cz = HKMath_cosf(((HKF32)HK_M_DEG_2_RAD) * rz * 0.5f); HKF32 sz = HKMath_sinf(((HKF32)HK_M_DEG_2_RAD) * rz * 0.5f);
	return HK_TYPE_INITIALIZER(HKQuat, -sx * sy * sz + cx * cy * cz, -cx * sy * sz + sx * cy * cz, cx * sy * cz + sx * cy * sz, sx * sy * cz + cx * cy * sz);
}
HK_INLINE                    HKQuat HKQuat_mat3x3_stack(
	HKF32 m00, HKF32 m10, HKF32 m20,
	HKF32 m01, HKF32 m11, HKF32 m21,
	HKF32 m02, HKF32 m12, HKF32 m22) HK_CXX_NOEXCEPT {
	HKF32 qx_base = (m00 - m11 - m22 + 1.0f);
	if (qx_base >= HK_MATH_QUAT_EQUAL_EPS) {
		HKF32 qx_2 = HKMath_sqrtf(qx_base);
		HKF32 qx = qx_2 / 2.0f;
		HKF32 qy = (m10 + m01) / (2.0f * qx_2);
		HKF32 qz = (m20 + m02) / (2.0f * qx_2);
		HKF32 qw = (m21 - m12) / (2.0f * qx_2);
		return HK_TYPE_INITIALIZER(HKQuat, qw, qx, qy, qz);
	}
	HKF32 qy_base = (-m00 + m11 - m22 + 1.0f);
	if (qy_base >= HK_MATH_QUAT_EQUAL_EPS) {
		HKF32 qy_2 = HKMath_sqrtf(qy_base);
		HKF32 qy = qy_2 / 2.0f;
		HKF32 qx = (m10 + m01) / (2.0f * qy_2);
		HKF32 qw = (m02 - m20) / (2.0f * qy_2);
		HKF32 qz = (m21 + m12) / (2.0f * qy_2);
		return HK_TYPE_INITIALIZER(HKQuat, qw, qx, qy, qz);
	}
	HKF32 qz_base = (-m00 - m11 + m22 + 1.0f);
	if (qz_base >= HK_MATH_QUAT_EQUAL_EPS) {
		HKF32 qz_2 = HKMath_sqrtf(qz_base);
		HKF32 qz = qz_2 / 2.0f;
		HKF32 qw = (m10 - m01) / (2.0f * qz_2);
		HKF32 qx = (m02 + m20) / (2.0f * qz_2);
		HKF32 qy = (m21 + m12) / (2.0f * qz_2);
		return HK_TYPE_INITIALIZER(HKQuat, qw, qx, qy, qz);
	}
	HKF32 qw_base = (m00 + m11 + m22 + 1.0f);
	if (qw_base >= HK_MATH_QUAT_EQUAL_EPS) {
		HKF32 qw_2 = HKMath_sqrtf(qw_base);
		HKF32 qw = qw_2 / 2.0f;
		HKF32 qz = (m10 - m01) / (2.0f * qw_2);
		HKF32 qy = (m02 - m20) / (2.0f * qw_2);
		HKF32 qx = (m21 - m12) / (2.0f * qw_2);
		return HK_TYPE_INITIALIZER(HKQuat, qw, qx, qy, qz);
	}
	return HK_TYPE_INITIALIZER(HKQuat, 0.0f, 0.0f, 0.0f, 0.0f);
}
HK_INLINE                    HKQuat HKQuat_mat3x3(const HKMat3x3* m) HK_CXX_NOEXCEPT {
	return HKQuat_mat3x3_stack(HK_MATH_MAT3X3_PASS_ARGUMENT_TO_STACK((*m)));
}
HK_INLINE                    HKQuat HKQuat_mat4x4(const HKMat4x4* m) HK_CXX_NOEXCEPT {
	return HKQuat_mat3x3_stack(HK_MATH_MAT3X3_PASS_ARGUMENT_TO_STACK((*m)));
}
#if defined(__cplusplus)
HK_INLINE HK_CXX11_CONSTEXPR HKBool  operator==(const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return  HKQuat_equal(&q1, &q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool  operator!=(const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return !HKQuat_equal(&q1, &q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator+ (const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return HKQuat_add(&q1,&q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator- (const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return HKQuat_sub(&q1,&q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator* (const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return HKQuat_mul(&q1,&q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator/ (const HKQuat& q1, const HKQuat& q2) HK_CXX_NOEXCEPT { return HKQuat_div(&q1,&q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator* (HKF32         v1, const HKQuat& q2) HK_CXX_NOEXCEPT { return HKQuat_mul_s1(v1, &q2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator* (const HKQuat& q1, HKF32         v2) HK_CXX_NOEXCEPT { return HKQuat_mul_s2(&q1, v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3  operator* (const HKQuat& q1, const HKVec3& v2) HK_CXX_NOEXCEPT { return HKQuat_rotateVector(&q1,&v2); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat  operator/ (const HKQuat& q1, HKF32         v2) HK_CXX_NOEXCEPT { return HKQuat_div_s2(&q1, v2); }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator= (const HKQuat& q) HK_CXX_NOEXCEPT { if (this != &q) { HKQuat_assign(this,&q); } return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator= (const HKCQuat& q)  HK_CXX_NOEXCEPT { w = q.w; x = q.x; y = q.y; z = q.z; return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator+=(const HKQuat& q) HK_CXX_NOEXCEPT { HKQuat_assignAdd(this, &q); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator-=(const HKQuat& q) HK_CXX_NOEXCEPT { HKQuat_assignSub(this, &q); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator*=(const HKQuat& q) HK_CXX_NOEXCEPT { HKQuat_assignMul(this, &q); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator*=(HKF32 v) HK_CXX_NOEXCEPT { HKQuat_assignMul_s(this, v); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKQuat& HKQuat::operator/=(HKF32 v) HK_CXX_NOEXCEPT { HKQuat_assignDiv_s(this, v); return *this; }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKQuat::rotateVector(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKQuat_rotateVector(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKQuat::lengthSqr()const HK_CXX_NOEXCEPT { return HKQuat_lengthSqr(this); }
HK_INLINE                    HKF32  HKQuat::length()const HK_CXX_NOEXCEPT { return HKQuat_length(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKF32  HKQuat::distanceSqr(const HKQuat& q)const HK_CXX_NOEXCEPT { return HKQuat_distanceSqr(this,&q); }
HK_INLINE                    HKF32  HKQuat::distance(const HKQuat& q)const HK_CXX_NOEXCEPT { return HKQuat_distance(this, &q); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::inverse()const HK_CXX_NOEXCEPT { return HKQuat_inverse(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::conjugate()const HK_CXX_NOEXCEPT { return HKQuat_conjugate(this); }
HK_INLINE                    HKQuat HKQuat::normalize()const HK_CXX_NOEXCEPT { return HKQuat_normalize(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::cross(const HKQuat& q)const HK_CXX_NOEXCEPT { return HKQuat_cross(this, &q); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::dot(const HKQuat& q)const HK_CXX_NOEXCEPT { return HKQuat_dot(this, &q); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat3x3 HKQuat::toMat3x3() const HK_CXX_NOEXCEPT { return HKQuat_toMat3x3(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 HKQuat::toMat4x4() const HK_CXX_NOEXCEPT { return HKQuat_toMat4x4(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::identity() HK_CXX_NOEXCEPT { return HKQuat_identity(); }
HK_INLINE HK_CXX11_CONSTEXPR HKQuat HKQuat::zeros() HK_CXX_NOEXCEPT { return HKQuat_zeros(); }
HK_INLINE                    HKQuat HKQuat::euler(HKF32 rx, HKF32 ry, HKF32 rz) HK_CXX_NOEXCEPT { return HKQuat_euler(rx, ry, rz); }
HK_INLINE                    HKVec3 HKQuat::eulerAngles() const HK_CXX_NOEXCEPT { return HKQuat_eulerAngles(this); }
HK_INLINE                    HKQuat HKQuat::mat3x3(const HKMat3x3& m) HK_CXX_NOEXCEPT { return HKQuat_mat3x3(&m); }
HK_INLINE                    HKQuat HKQuat::mat4x4(const HKMat4x4& m) HK_CXX_NOEXCEPT { return HKQuat_mat4x4(&m); }
#endif
#endif
