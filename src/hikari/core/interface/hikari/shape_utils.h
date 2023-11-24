#ifndef HK_CORE_SHAPE_UTILS__H
#define HK_CORE_SHAPE_UTILS__H
#include "object.h"

#if defined(__cplusplus)
#define HK_SHAPE_ARRAY_DEFINE(TYPE)                                                             \
    struct HKArray##TYPE : public HKShape                                                       \
    {                                                                                           \
        HK_OBJECT_ARRAY_METHOD_DEFINE(TYPE);                                                    \
    };                                                                                          \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE);                                                        \
    HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(HKArray##TYPE,HKShape,getAabb,HKCAabb);              \
    HK_OBJECT_CREATE_TRAITS(HKArray##TYPE);                                                     \
    template<>  struct HKArrayTraits<HK##TYPE> { typedef HKArray##TYPE type; }
#else
#define HK_SHAPE_ARRAY_DEFINE(TYPE)             \
    typedef struct HKArray##TYPE HKArray##TYPE; \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE);        \
    HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(HKArray##TYPE, HKShape,getAabb, HKCAabb)
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

#define HK_SHAPE_C_DERIVE_METHODS(TYPE) \
HK_OBJECT_C_DERIVE_METHODS(TYPE); \
HK_OBJECT_C_DERIVE_METHOD_DECL_1_CONST(TYPE,HKShape,getAabb,HKCAabb)

#endif
