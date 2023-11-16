#ifndef HK_SHAPE__H
#define HK_SHAPE__H

#include "object.h"
#include "object_array.h"
#include "math/aabb.h"

#define HK_OBJECT_TYPEID_Shape HK_UUID_DEFINE(0x4cc662fc, 0xad12, 0x4ba6, 0xa2, 0x93, 0xc7, 0xa9, 0x56, 0x75, 0xfa, 0xe2)
#define HK_OBJECT_TYPEID_ArrayShape HK_UUID_DEFINE(0xa51f3e36, 0x59a3, 0x4c5c, 0x84, 0x2f, 0x7e, 0xa5, 0xb3, 0x6a, 0xa7, 0x40)

#if defined(__cplusplus)
struct HKShape : public HKUnknown
{
    static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT { return HK_OBJECT_TYPEID_Shape; }
    virtual HKAabb HK_API getAabb() const = 0;
};
#else
typedef struct HKShape HKShape;
#endif

HK_EXTERN_C HK_DLL HKCAabb HK_API HKShape_getAabb(const HKShape *shape);

#if defined(__cplusplus)
#define HK_SHAPE_ARRAY_DEFINE(TYPE)                                                             \
    struct HKArray##TYPE : public HKShape                                                       \
    {                                                                                           \
        HK_OBJECT_ARRAY_METHOD_DEFINE(TYPE);                                                    \
    };                                                                                          \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE);                                                        \
    HK_INLINE HKArray##TYPE *HKArray##TYPE::create() { return HKArray##TYPE##_create(); }

#else
#define HK_SHAPE_ARRAY_DEFINE(TYPE)            \
    typedef struct HKArray##TYPE HKArray##TYPE; \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE)
#endif

#define HK_SHAPE_ARRAY_IMPL_DEFINE(TYPE)                                                                                 \
    struct HK_DLL HKArray##TYPE##Impl : public HKArray##TYPE, protected HKRefCntObject                                   \
    {                                                                                                                    \
        HKArray##TYPE##Impl() : HKArray##TYPE(), HKRefCntObject(), arr{} {}                                              \
        HKArray##TYPE##Impl(const std::vector<HK##TYPE *> &arr_) : HKArray##TYPE(), HKRefCntObject(), arr{arr_} {}       \
        virtual HK_API ~HKArray##TYPE##Impl() {}                                                                         \
        virtual HKU32 HK_API addRef() override { return HKRefCntObject::addRef(); }                                      \
        virtual HKU32 HK_API release() override { return HKRefCntObject::release(); }                                    \
        virtual HKBool HK_API queryInterface(HKUUID iid, void **ppvInterface) override                                   \
        {                                                                                                                \
            if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_Array##TYPE) \
            {                                                                                                            \
                addRef();                                                                                                \
                *ppvInterface = this;                                                                                    \
                return true;                                                                                             \
            }                                                                                                            \
            else                                                                                                         \
            {                                                                                                            \
                return false;                                                                                            \
            }                                                                                                            \
        }                                                                                                                \
        virtual HKAabb HK_API getAabb() const override                                                                   \
        {                                                                                                                \
            HKAabb aabb;                                                                                                 \
            for (auto &shape : arr)                                                                                      \
            {                                                                                                            \
                if (shape)                                                                                               \
                {                                                                                                        \
                    aabb = aabb | shape->getAabb();                                                                      \
                }                                                                                                        \
            }                                                                                                            \
            return aabb;                                                                                                 \
        }                                                                                                                \
        HK_OBJECT_ARRAY_METHOD_IMPL_DEFINE(TYPE);                                                                        \
    };                                                                                                                   \
    HK_OBJECT_ARRAY_IMPL_DEFINE_COMMON(TYPE)

HK_SHAPE_ARRAY_DEFINE(Shape);

#endif
