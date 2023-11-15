#ifndef HK_MATH_AFFINE_3D__H
#define HK_MATH_AFFINE_3D__H

#include "vec.h"
#include "quat.h"
#include "matrix.h"
// Affine3D��POD�ł���ׂ�����, ���̂܂܂ł�POD�^�ɂȂ�Ȃ�
// ������POD�p�ɕʂ̃N���X���`����
typedef struct HKCAffine3D {
    HKCQuat rotation;
    HKCVec3 scaling;
    HKCVec3 position;
} HKCAffine3D;

#if defined(__cplusplus)
typedef struct HKAffine3D {
    HK_CXX11_CONSTEXPR HKAffine3D() HK_CXX_NOEXCEPT : rotation{ 1.0f }, scaling{ 1.0f }, position{ 0.0f } {}
    HK_CXX11_CONSTEXPR HKAffine3D(const HKQuat& r, const HKVec3& s, const HKVec3& p) HK_CXX_NOEXCEPT : rotation{ r }, scaling{ s }, position{ p } {}
    HK_CXX11_CONSTEXPR HKAffine3D(const HKAffine3D& af) HK_CXX_NOEXCEPT   : rotation{ af.rotation }, scaling{ af.scaling }, position{ af.position } {}
    HK_CXX11_CONSTEXPR HKAffine3D(const HKCAffine3D& af) HK_CXX_NOEXCEPT : rotation{ af.rotation }, scaling{ af.scaling }, position{ af.position } {}
    HK_CXX14_CONSTEXPR HKAffine3D& operator=(const HKAffine3D& af) HK_CXX_NOEXCEPT { if (this != &af) { rotation = af.rotation; scaling = af.scaling; position = af.position; } return *this;  }
    HK_CXX14_CONSTEXPR HKAffine3D& operator=(const HKCAffine3D& af) HK_CXX_NOEXCEPT { { rotation = af.rotation; scaling = af.scaling; position = af.position; } return *this; }
    HK_INLINE HK_CXX14_CONSTEXPR HKAffine3D& operator*=(const HKAffine3D&  af) HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   operator*(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR operator HKCAffine3D() const HK_CXX_NOEXCEPT { return HKCAffine3D{ (HKCQuat)rotation,(HKCVec3)scaling,(HKCVec3)position }; }
    HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 toMat4x4() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   transformPoint    (const HKVec3& p) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   transformVector   (const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   transformDirection(const HKVec3& d) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   inverseTransformPoint(const HKVec3& p) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   inverseTransformVector(const HKVec3& v) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3   inverseTransformDirection(const HKVec3& d) const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 transformPointMatrix()  const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4 transformVectorMatrix() const HK_CXX_NOEXCEPT;
    static HK_INLINE HKAffine3D mat4x4(const HKMat4x4& m) HK_CXX_NOEXCEPT;

    HKQuat rotation;
    HKVec3 scaling ;
    HKVec3 position;
} HKAffine3D;
#else
// C�ł�CAffine3D��Affine3D�Ƃ��Ĉ���
typedef struct HKCAffine3D HKAffine3D;
#endif
HK_INLINE HK_CXX11_CONSTEXPR HKBool     HKAffine3D_equal_withEps(const HKAffine3D* af1, const HKAffine3D* af2, HKF32 eps) HK_CXX_NOEXCEPT {
    return HKVec3_equal_withEps(&af1->position, &af2->position, eps) && HKVec3_equal_withEps(&af1->scaling, &af2->scaling, eps) && HKQuat_equal_withEps(&af1->rotation, &af2->rotation, eps);
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool     HKAffine3D_equal(const HKAffine3D* af1, const HKAffine3D* af2) HK_CXX_NOEXCEPT {
    return HKVec3_equal(&af1->position, &af2->position) && HKVec3_equal(&af1->scaling, &af2->scaling) && HKQuat_equal(&af1->rotation, &af2->rotation);
}
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4   HKAffine3D_toMat4x4(const HKAffine3D* af) HK_CXX_NOEXCEPT {
#if defined(__cplusplus)
    return HKMat4x4::translate(af->position) * af->rotation.toMat4x4() * HKMat4x4::scale(af->scaling);
#else
    HKMat4x4 t = HKMat4x4_translate_3(&af->position);
    HKMat4x4 s = HKMat4x4_scale_3(&af->scaling);
    HKMat4x4 r = HKQuat_toMat4x4(&af->rotation);
    HKMat4x4 m = HKMat4x4_mul(&r, &s);
    return HKMat4x4_mul(&t, &m);
#endif
}
HK_INLINE                    HKAffine3D HKAffine3D_mat4x4(const HKMat4x4* m) HK_CXX_NOEXCEPT {
    // translate
    HKAffine3D af;
    af.position.x = m->c3.x;// af.x
    af.position.y = m->c3.y;// af.y
    af.position.z = m->c3.z;// af.z
    // scaling
    HKF32  s = HKMath_cbrtf(HKMat4x4_determinant(m));
    af.scaling.x = s;
    af.scaling.y = s;
    af.scaling.z = s;
    // rotation
    af.rotation = HKQuat_mat3x3_stack(
        m->c0.x / s, m->c0.y / s, m->c0.z / s,
        m->c1.x / s, m->c1.y / s, m->c1.z / s,
        m->c2.x / s, m->c2.y / s, m->c2.z / s
    );
    return af;
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_mulVector(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKVec3_add_stack_2((&af1->position), HKMat3x3_mulVector_v2_stack(HKQuat_toMat3x3(&af1->rotation), HK_MATH_VEC3_ARITHMETRIC_INIT(af1->scaling,*,(*v2))));
}
HK_INLINE HK_CXX11_CONSTEXPR HKAffine3D HKAffine3D_mul(const HKAffine3D* af1, const HKAffine3D*  af2) HK_CXX_NOEXCEPT {
    return HK_TYPE_INITIALIZER(HKAffine3D, 
        HKQuat_mul((&af1->rotation), (&af2->rotation)),
        HKVec3_mul((&af1->scaling) , (&af2->scaling )),
        HKAffine3D_mulVector(af1, &(af2->position))
    );
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_transformPoint(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKAffine3D_mulVector(af1, v2);
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_transformVector(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKMat3x3_mulVector_v2_stack(HKQuat_toMat3x3(&af1->rotation), HK_MATH_VEC3_ARITHMETRIC_INIT(af1->scaling, *, (*v2)));
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_transformDirection(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKQuat_rotateVector(&af1->rotation, v2);
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_inverseTransformPoint(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKVec3_div_stack_1(HKMat3x3_mulVector_v1_stack(HKVec3_sub(v2, &af1->position), HKQuat_toMat3x3(&af1->rotation)), &(af1->scaling));
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_inverseTransformVector(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKVec3_div_stack_1(HKMat3x3_mulVector_v1_stack_m2(v2, HKQuat_toMat3x3(&af1->rotation)), &(af1->scaling));
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3     HKAffine3D_inverseTransformDirection(const HKAffine3D* af1, const HKVec3* v2) HK_CXX_NOEXCEPT {
    return HKMat3x3_mulVector_v1_stack_m2(v2, HKQuat_toMat3x3(&af1->rotation));
}
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4   HKAffine3D_transformPointMatrix(const HKAffine3D* af) HK_CXX_NOEXCEPT {
    return HKAffine3D_toMat4x4(af);
}
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4   HKAffine3D_transformVectorMatrix(const HKAffine3D* af) HK_CXX_NOEXCEPT {
#if defined(__cplusplus)
    return af->rotation.toMat4x4() * HKMat4x4::scale(af->scaling);
#else    
    HKMat4x4 s = HKMat4x4_scale_3(&af->scaling);
    HKMat4x4 r = HKQuat_toMat4x4(&af->rotation);
    return HKMat4x4_mul(&r, &s);
#endif
}
#if defined(__cplusplus)
HK_INLINE HK_CXX11_CONSTEXPR HKBool      operator==(const HKAffine3D& af1, const HKAffine3D& af2) HK_CXX_NOEXCEPT { return  HKAffine3D_equal(&af1, &af2); }
HK_INLINE HK_CXX11_CONSTEXPR HKBool      operator!=(const HKAffine3D& af1, const HKAffine3D& af2) HK_CXX_NOEXCEPT { return !HKAffine3D_equal(&af1, &af2); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4    HKAffine3D::toMat4x4() const HK_CXX_NOEXCEPT { return HKAffine3D_toMat4x4(this); }
HK_INLINE                    HKAffine3D  HKAffine3D::mat4x4(const HKMat4x4& m) HK_CXX_NOEXCEPT { return HKAffine3D_mat4x4(&m); }
HK_INLINE HK_CXX14_CONSTEXPR HKAffine3D& HKAffine3D::operator*=(const HKAffine3D& af) HK_CXX_NOEXCEPT { *this = HKAffine3D_mul(this, &af); return *this; }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::operator*(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKAffine3D_mulVector(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::transformPoint(const HKVec3& p) const HK_CXX_NOEXCEPT { return HKAffine3D_transformPoint(this,&p); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::transformVector(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKAffine3D_transformVector(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::transformDirection(const HKVec3& d) const HK_CXX_NOEXCEPT { return HKAffine3D_transformDirection(this, &d); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::inverseTransformPoint(const HKVec3& p) const HK_CXX_NOEXCEPT { return HKAffine3D_inverseTransformPoint(this, &p); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::inverseTransformVector(const HKVec3& v) const HK_CXX_NOEXCEPT { return HKAffine3D_inverseTransformVector(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3      HKAffine3D::inverseTransformDirection(const HKVec3& d) const HK_CXX_NOEXCEPT { return HKAffine3D_inverseTransformDirection(this, &d); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4    HKAffine3D::transformPointMatrix()  const HK_CXX_NOEXCEPT { return HKAffine3D_transformPointMatrix(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKMat4x4    HKAffine3D::transformVectorMatrix() const HK_CXX_NOEXCEPT { return HKAffine3D_transformVectorMatrix(this); }

#endif

#endif
