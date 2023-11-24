#ifndef HK_MATH_AABB__H
#define HK_MATH_AABB__H

#include "vec.h"

typedef struct HKCAabb {
    HKCVec3 min;
    HKCVec3 max;
} HKCAabb;

#if defined(__cplusplus)
typedef struct HKAabb {
    HK_CXX11_CONSTEXPR HKAabb() HK_CXX_NOEXCEPT : min{HK_FLT_MAX,HK_FLT_MAX,HK_FLT_MAX },max{-HK_FLT_MAX,-HK_FLT_MAX,-HK_FLT_MAX }{}
    HK_CXX11_CONSTEXPR HKAabb(const HKVec3& min_, const HKVec3& max_) HK_CXX_NOEXCEPT : min{min_},max{max_}{}
    HK_CXX11_CONSTEXPR HKAabb(const HKAabb& aabb) HK_CXX_NOEXCEPT : min{aabb.min},max{aabb.max}{}
    HK_CXX11_CONSTEXPR HKAabb(const HKCAabb& aabb) HK_CXX_NOEXCEPT : min{HKCVec3(aabb.min)},max{HKCVec3(aabb.max)}{}
    HK_CXX14_CONSTEXPR HKAabb& operator=(const HKAabb& aabb) HK_CXX_NOEXCEPT { if (this!=&aabb){ min = aabb.min; max = aabb.max; } return *this; }
    HK_CXX14_CONSTEXPR HKAabb& operator=(const HKCAabb& aabb) HK_CXX_NOEXCEPT { { min = HKCVec3(aabb.min); max = HKCVec3(aabb.max); } return *this; }
    HK_INLINE HK_CXX11_CONSTEXPR operator HKCAabb() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCAabb, HKCVec3(min), HKCVec3(max)); }
    HK_INLINE HK_CXX14_CONSTEXPR HKAabb& operator|=(const HKAabb&  aabb) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR HKAabb& operator&=(const HKAabb&  aabb) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKBool containPoint(const HKVec3& v) const  HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKAabb addPoint(const HKVec3& p) const  HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3 getMax() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3 getMin() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3 getCenter() const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec3 getRange () const HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR void   setMax(const HKVec3& max) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR void   setMin(const HKVec3& min) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR void   setCenter(const HKVec3& c) HK_CXX_NOEXCEPT;
    HK_INLINE HK_CXX14_CONSTEXPR void   setRange (const HKVec3& r) HK_CXX_NOEXCEPT;

    HKVec3 min;
    HKVec3 max;
} HKAabb;
#else
typedef struct HKCAabb HKAabb;
#endif
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKAabb_equal_withEps(const HKAabb*  aabb1,const HKAabb*  aabb2, HKF32 eps) HK_CXX_NOEXCEPT {
    return HKVec3_equal_withEps(&aabb1->min,&aabb2->min,eps) &&HKVec3_equal_withEps(&aabb1->max,&aabb2->max,eps);
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKAabb_equal(const HKAabb*  aabb1,const HKAabb*  aabb2) HK_CXX_NOEXCEPT {
    return HKVec3_equal(&aabb1->min,&aabb2->min) &&HKVec3_equal(&aabb1->max,&aabb2->max);
}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb HKAabb_create() { return HK_TYPE_INITIALIZER(HKAabb,HKVec3_create1(HK_FLT_MAX),HKVec3_create1(-HK_FLT_MAX));}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb HKAabb_create2(HKVec3 min, HKVec3 max) { return HK_TYPE_INITIALIZER(HKAabb,min,max);}
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_assignAnd(HKAabb*  aabb1,const HKAabb*  aabb2) HK_CXX_NOEXCEPT {
    aabb1->min.x = HKMath_fmaxf(aabb1->min.x,aabb2->min.x);
    aabb1->min.y = HKMath_fmaxf(aabb1->min.y,aabb2->min.y);
    aabb1->min.z = HKMath_fmaxf(aabb1->min.z,aabb2->min.z);
    aabb1->max.x = HKMath_fminf(aabb1->max.x,aabb2->max.x);
    aabb1->max.y = HKMath_fminf(aabb1->max.y,aabb2->max.y);
    aabb1->max.z = HKMath_fminf(aabb1->max.z,aabb2->max.z);
}
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_assignOr (HKAabb*  aabb1,const HKAabb*  aabb2) HK_CXX_NOEXCEPT {
    aabb1->min.x = HKMath_fminf(aabb1->min.x,aabb2->min.x);
    aabb1->min.y = HKMath_fminf(aabb1->min.y,aabb2->min.y);
    aabb1->min.z = HKMath_fminf(aabb1->min.z,aabb2->min.z);
    aabb1->max.x = HKMath_fmaxf(aabb1->max.x,aabb2->max.x);
    aabb1->max.y = HKMath_fmaxf(aabb1->max.y,aabb2->max.y);
    aabb1->max.z = HKMath_fmaxf(aabb1->max.z,aabb2->max.z);
}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb HKAabb_and(const HKAabb*  aabb1,const HKAabb*  aabb2) HK_CXX_NOEXCEPT { 
    return HK_TYPE_INITIALIZER(HKAabb,
        HK_TYPE_INITIALIZER(HKVec3,
            HKMath_fmaxf(aabb1->min.x,aabb2->min.x),
            HKMath_fmaxf(aabb1->min.y,aabb2->min.y),
            HKMath_fmaxf(aabb1->min.z,aabb2->min.z)
        ),
        HK_TYPE_INITIALIZER(HKVec3,
            HKMath_fminf(aabb1->max.x,aabb2->max.x),
            HKMath_fminf(aabb1->max.y,aabb2->max.y),
            HKMath_fminf(aabb1->max.z,aabb2->max.z)
        )
    );
}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb HKAabb_or (const HKAabb*  aabb1,const HKAabb*  aabb2) HK_CXX_NOEXCEPT {
    return HK_TYPE_INITIALIZER(HKAabb,
        HK_TYPE_INITIALIZER(HKVec3,
            HKMath_fminf(aabb1->min.x,aabb2->min.x),
            HKMath_fminf(aabb1->min.y,aabb2->min.y),
            HKMath_fminf(aabb1->min.z,aabb2->min.z)
        ),
        HK_TYPE_INITIALIZER(HKVec3,
            HKMath_fmaxf(aabb1->max.x,aabb2->max.x),
            HKMath_fmaxf(aabb1->max.y,aabb2->max.y),
            HKMath_fmaxf(aabb1->max.z,aabb2->max.z)
        )
    );
}
HK_INLINE HK_CXX11_CONSTEXPR HKBool HKAabb_containPoint(const HKAabb* aabb, const HKVec3*   v) HK_CXX_NOEXCEPT {
    return 
    (HKMath_fclampf(v->x,aabb->min.x,aabb->max.x)==v->x)&& 
    (HKMath_fclampf(v->y,aabb->min.y,aabb->max.y)==v->y)&& 
    (HKMath_fclampf(v->z,aabb->min.z,aabb->max.z)==v->z);
}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb HKAabb_addPoint(const HKAabb* aabb, const HKVec3* p)  HK_CXX_NOEXCEPT {
    {   
        return HK_TYPE_INITIALIZER(HKAabb, 
            HK_TYPE_INITIALIZER(HKVec3, HKMath_fminf(aabb->min.x, p->x), HKMath_fminf(aabb->min.y, p->y), HKMath_fminf(aabb->min.z, p->z)),
            HK_TYPE_INITIALIZER(HKVec3, HKMath_fmaxf(aabb->max.x, p->x), HKMath_fmaxf(aabb->max.y, p->y), HKMath_fmaxf(aabb->max.z, p->z))
        );
    }
}
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb_getMax(const HKAabb* aabb) HK_CXX_NOEXCEPT { return aabb->max; }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb_getMin(const HKAabb* aabb) HK_CXX_NOEXCEPT { return aabb->min; }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb_getCenter(const HKAabb* aabb) HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, (0.5f*(aabb->min.x+ aabb->max.x)), (0.5f * (aabb->min.y + aabb->max.y)), (0.5f * (aabb->min.z + aabb->max.z))); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb_getRange(const HKAabb* aabb)  HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKVec3, ((aabb->max.x - aabb->min.x)), ((aabb->max.y - aabb->min.y)), ((aabb->max.z - aabb->min.z))); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_setMax(HKAabb* aabb, const HKVec3* max) HK_CXX_NOEXCEPT { HK_MATH_VEC3_ARITHMETRIC_ASSIGN((aabb->max), =, (*max)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_setMin(HKAabb* aabb, const HKVec3* min) HK_CXX_NOEXCEPT { HK_MATH_VEC3_ARITHMETRIC_ASSIGN((aabb->min), =, (*min)); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_setCenter(HKAabb* aabb, const HKVec3* c) HK_CXX_NOEXCEPT { 
    HKVec3 range = HKAabb_getRange(aabb); 
    aabb->max.x = c->x + range.x * 0.5f; 
    aabb->min.x = c->x - range.x * 0.5f;
    aabb->max.y = c->y + range.y * 0.5f;
    aabb->min.y = c->y - range.y * 0.5f;
    aabb->max.z = c->z + range.z * 0.5f;
    aabb->min.z = c->z - range.z * 0.5f;
}
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb_setRange( HKAabb* aabb, const HKVec3* r) HK_CXX_NOEXCEPT {
    HKVec3 center = HKAabb_getCenter(aabb); 
    aabb->max.x = center.x + r->x * 0.5f;
    aabb->min.x = center.x - r->x * 0.5f;
    aabb->max.y = center.y + r->y * 0.5f;
    aabb->min.y = center.y - r->y * 0.5f;
    aabb->max.z = center.z + r->z * 0.5f;
    aabb->min.z = center.z - r->z * 0.5f;
}
#if defined(__cplusplus)
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator==(const HKAabb&  aabb1,const HKAabb&  aabb2)HK_CXX_NOEXCEPT { return  HKAabb_equal(&aabb1,&aabb2);}
HK_INLINE HK_CXX11_CONSTEXPR HKBool operator!=(const HKAabb&  aabb1,const HKAabb&  aabb2)HK_CXX_NOEXCEPT { return !HKAabb_equal(&aabb1,&aabb2);}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb operator|(const HKAabb&  aabb1,const HKAabb&  aabb2) HK_CXX_NOEXCEPT { 
    return HKAabb_or(&aabb1,&aabb2);
}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb operator&(const HKAabb&  aabb1,const HKAabb&  aabb2) HK_CXX_NOEXCEPT { 
    return HKAabb_and(&aabb1,&aabb2);
 }
HK_INLINE HK_CXX14_CONSTEXPR HKAabb& HKAabb::operator|=(const HKAabb&  aabb) HK_CXX_NOEXCEPT { HKAabb_assignOr(this,&aabb); return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKAabb& HKAabb::operator&=(const HKAabb&  aabb) HK_CXX_NOEXCEPT{ HKAabb_assignAnd(this,&aabb); return *this; }
HK_INLINE HK_CXX11_CONSTEXPR HKBool  HKAabb::containPoint(const HKVec3& v) const  HK_CXX_NOEXCEPT{ return HKAabb_containPoint(this,&v);}
HK_INLINE HK_CXX11_CONSTEXPR HKAabb  HKAabb::addPoint(const HKVec3& v) const  HK_CXX_NOEXCEPT { return HKAabb_addPoint(this, &v); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb::getMax() const HK_CXX_NOEXCEPT { return HKAabb_getMax(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb::getMin() const HK_CXX_NOEXCEPT { return HKAabb_getMin(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb::getCenter() const HK_CXX_NOEXCEPT { return HKAabb_getCenter(this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec3 HKAabb::getRange() const HK_CXX_NOEXCEPT  { return HKAabb_getRange (this); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb::setMax(const HKVec3& max) HK_CXX_NOEXCEPT { return HKAabb_setMax(this,&max); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb::setMin(const HKVec3& min) HK_CXX_NOEXCEPT { return HKAabb_setMin(this,&min); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb::setCenter(const HKVec3& c) HK_CXX_NOEXCEPT{ return HKAabb_setCenter(this, &c); }
HK_INLINE HK_CXX14_CONSTEXPR void   HKAabb::setRange(const HKVec3& r) HK_CXX_NOEXCEPT { return HKAabb_setRange(this , &r); }
#endif

#endif
