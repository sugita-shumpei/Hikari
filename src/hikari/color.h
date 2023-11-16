#ifndef HK_COLOR__H
#define HK_COLOR__H

#include "data_type.h"
#include "math/vec.h"

typedef struct HKCColor {    
    HKF32 r;
    HKF32 g;
    HKF32 b;
    HKF32 a;
} HKCColor;

#if defined(__cplusplus)
struct         HKColor8;
typedef struct HKColor {    
    HK_CXX11_CONSTEXPR HKColor() HK_CXX_NOEXCEPT : r{},g{},b{},a{}{}
    HK_CXX11_CONSTEXPR HKColor(HKF32 r_,HKF32 g_,HKF32 b_,HKF32 a_) HK_CXX_NOEXCEPT : r{r_},g{g_},b{b_},a{a_}{}
    HK_CXX11_CONSTEXPR HKColor(const HKColor&  c) HK_CXX_NOEXCEPT : r{c.r},g{c.g},b{c.b},a{c.a}{}
    HK_CXX11_CONSTEXPR HKColor(const HKCColor& c) HK_CXX_NOEXCEPT : r{c.r},g{c.g},b{c.b},a{c.a}{}
    HK_INLINE HK_CXX11_CONSTEXPR HKColor(const HKColor8& c) HK_CXX_NOEXCEPT ;
    HK_CXX11_CONSTEXPR HKColor(const HKVec4  & v) HK_CXX_NOEXCEPT : r{v.x},g{v.y},b{v.z},a{v.w}{}

    HK_INLINE HK_CXX14_CONSTEXPR HKColor& operator=(const HKColor&  c) HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX14_CONSTEXPR HKColor& operator=(const HKCColor& c) HK_CXX_NOEXCEPT ;
    
    HK_INLINE HK_CXX11_CONSTEXPR operator HKCColor() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCColor,r,g,b,a);}
    HK_INLINE HK_CXX11_CONSTEXPR operator HKColor8() const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR operator HKVec4  () const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR HKColor8 toColor8() const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec4   toVec4  () const HK_CXX_NOEXCEPT ;
    HK_INLINE                    HKColor  toGamma () const HK_CXX_NOEXCEPT ;
    HK_INLINE                    HKColor  toLinear() const HK_CXX_NOEXCEPT ;

    HKF32 r;
    HKF32 g;
    HKF32 b;
    HKF32 a;
} HKColor;
#else
typedef struct HKCColor HKColor;
#endif

typedef struct HKCColor8 {
    HKU8 r;
    HKU8 g;
    HKU8 b;
    HKU8 a;
} HKCColor8;

#if defined(__cplusplus)
typedef struct HKColor8 {    
    HK_CXX11_CONSTEXPR HKColor8() HK_CXX_NOEXCEPT : r{},g{},b{},a{}{}
    HK_CXX11_CONSTEXPR HKColor8(HKU8 r_,HKU8 g_,HKU8 b_,HKU8 a_) HK_CXX_NOEXCEPT : r{r_},g{g_},b{b_},a{a_}{}
    HK_CXX11_CONSTEXPR HKColor8(const HKColor8 & c) HK_CXX_NOEXCEPT : r{c.r},g{c.g},b{c.b},a{c.a}{}
    HK_CXX11_CONSTEXPR HKColor8(const HKCColor8& c) HK_CXX_NOEXCEPT : r{c.r},g{c.g},b{c.b},a{c.a}{}
    HK_CXX11_CONSTEXPR HKColor8(const HKColor  & c) HK_CXX_NOEXCEPT : 
        r{static_cast<HKU8>(HKMath_fclampf(c.r*256.0f,0.0,255.0f))},
        g{static_cast<HKU8>(HKMath_fclampf(c.g*256.0f,0.0,255.0f))},
        b{static_cast<HKU8>(HKMath_fclampf(c.b*256.0f,0.0,255.0f))},
        a{static_cast<HKU8>(HKMath_fclampf(c.a*256.0f,0.0,255.0f))}
    {}
    HK_CXX11_CONSTEXPR HKColor8(const HKVec4   & v) HK_CXX_NOEXCEPT : 
        r{static_cast<HKU8>(HKMath_fclampf(v.x*256.0f,0.0,255.0f))},
        g{static_cast<HKU8>(HKMath_fclampf(v.y*256.0f,0.0,255.0f))},
        b{static_cast<HKU8>(HKMath_fclampf(v.z*256.0f,0.0,255.0f))},
        a{static_cast<HKU8>(HKMath_fclampf(v.w*256.0f,0.0,255.0f))}
    {}

    HK_INLINE HK_CXX14_CONSTEXPR HKColor8& operator=(const HKColor8&  c) HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX14_CONSTEXPR HKColor8& operator=(const HKCColor8& c) HK_CXX_NOEXCEPT ;
    
    HK_INLINE HK_CXX11_CONSTEXPR operator HKCColor8() const HK_CXX_NOEXCEPT { return HK_TYPE_INITIALIZER(HKCColor8,r,g,b,a);}
    HK_INLINE HK_CXX11_CONSTEXPR operator HKColor  () const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR operator HKVec4   () const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR HKColor  toColor  () const HK_CXX_NOEXCEPT ;
    HK_INLINE HK_CXX11_CONSTEXPR HKVec4   toVec4   () const HK_CXX_NOEXCEPT ;
    HK_INLINE                    HKColor8 toGamma () const HK_CXX_NOEXCEPT ;
    HK_INLINE                    HKColor8 toLinear() const HK_CXX_NOEXCEPT ;

    HKU8 r;
    HKU8 g;
    HKU8 b;
    HKU8 a;
} HKColor8;

HK_INLINE HK_CXX11_CONSTEXPR HKColor::HKColor(const HKColor8& c) HK_CXX_NOEXCEPT :
r{ static_cast<HKF32>(c.r) / 255.0f },
g{ static_cast<HKF32>(c.g) / 255.0f },
b{ static_cast<HKF32>(c.b) / 255.0f },
a{ static_cast<HKF32>(c.a) / 255.0f }
{}
#else
typedef struct HKCColor8 HKColor8;
#endif

#if defined(__cplusplus)
HK_INLINE HK_CXX14_CONSTEXPR HKColor&  HKColor ::operator=(const HKColor  & c) HK_CXX_NOEXCEPT { if (this!=&c){ r = c.r; g = c.g; b = c.b; a = c.a; } return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKColor&  HKColor ::operator=(const HKCColor & c) HK_CXX_NOEXCEPT {              { r = c.r; g = c.g; b = c.b; a = c.a; } return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKColor8& HKColor8::operator=(const HKColor8 & c) HK_CXX_NOEXCEPT { if (this!=&c){ r = c.r; g = c.g; b = c.b; a = c.a; } return *this; }
HK_INLINE HK_CXX14_CONSTEXPR HKColor8& HKColor8::operator=(const HKCColor8& c) HK_CXX_NOEXCEPT {              { r = c.r; g = c.g; b = c.b; a = c.a; } return *this; }

HK_INLINE HK_CXX11_CONSTEXPR HKColor ::operator HKColor8() const HK_CXX_NOEXCEPT { return HKColor8(*this); }
HK_INLINE HK_CXX11_CONSTEXPR HKColor ::operator HKVec4  () const HK_CXX_NOEXCEPT { return HKVec4(r,g,b,a); }
HK_INLINE HK_CXX11_CONSTEXPR HKColor8::operator HKColor () const HK_CXX_NOEXCEPT { return HKColor(*this); }
HK_INLINE HK_CXX11_CONSTEXPR HKColor8::operator HKVec4  () const HK_CXX_NOEXCEPT { return HKVec4(
    static_cast<HKF32>(r)/255.0f,
    static_cast<HKF32>(g)/255.0f,
    static_cast<HKF32>(b)/255.0f,
    static_cast<HKF32>(a)/255.0f); 
}

HK_INLINE HK_CXX11_CONSTEXPR HKColor8 HKColor ::toColor8() const HK_CXX_NOEXCEPT { return HKColor8(*this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4   HKColor ::toVec4  () const HK_CXX_NOEXCEPT { return HKVec4(r,g,b,a); }
HK_INLINE HK_CXX11_CONSTEXPR HKColor  HKColor8::toColor () const HK_CXX_NOEXCEPT { return HKColor(*this); }
HK_INLINE HK_CXX11_CONSTEXPR HKVec4   HKColor8::toVec4  () const HK_CXX_NOEXCEPT { return HKVec4(
    static_cast<HKF32>(r)/255.0f,
    static_cast<HKF32>(g)/255.0f,
    static_cast<HKF32>(b)/255.0f,
    static_cast<HKF32>(a)/255.0f); 
}



#endif

#endif
