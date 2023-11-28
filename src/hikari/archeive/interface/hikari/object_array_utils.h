#ifndef HK_OBJECT_ARRAY_UTILS__H
#define HK_OBJECT_ARRAY_UTILS__H

#include <hikari/plugin.h>

#define HK_OBJECT_ARRAY_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,TYPE) \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_create); \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_clone);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_getCapacity);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_setCapacity);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_getCount);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_setCount);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_setValue);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_internal_getValue_const);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_internal_getValue);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_internal_getPointer_const);  \
	HK_PLUGIN_DEFINE_GET_PROC_ADDRESS(NAME,HKArray##TYPE##_internal_getPointer)

#define HK_OBJECT_ARRAY_DEFINE_COMMON_DLL(TYPE)                                                                                                                   \
    HK_EXTERN_C HK_DLL_FUNCTION HKArray##TYPE*        HK_DLL_FUNCTION_NAME(HKArray##TYPE##_create)();                                                             \
    HK_EXTERN_C HK_DLL_FUNCTION HKArray##TYPE*        HK_DLL_FUNCTION_NAME(HKArray##TYPE##_clone )(const HKArray##TYPE *p);                                       \
    HK_EXTERN_C HK_DLL_FUNCTION HKU64                 HK_DLL_FUNCTION_NAME(HKArray##TYPE##_getCapacity)(const HKArray##TYPE *p);                                  \
    HK_EXTERN_C HK_DLL_FUNCTION void                  HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setCapacity)(HKArray##TYPE *p, HKU64 c);                               \
    HK_EXTERN_C HK_DLL_FUNCTION HKU64                 HK_DLL_FUNCTION_NAME(HKArray##TYPE##_getCount)(const HKArray##TYPE *p);                                     \
    HK_EXTERN_C HK_DLL_FUNCTION void                  HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setCount)(HKArray##TYPE *p, HKU64 c);                                  \
    HK_EXTERN_C HK_DLL_FUNCTION void                  HK_DLL_FUNCTION_NAME(HKArray##TYPE##_setValue)(HKArray##TYPE *p, HKU64 idx, HK##TYPE *v);                   \
    HK_EXTERN_C HK_DLL_FUNCTION const HK##TYPE*       HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getValue_const)(const HKArray##TYPE *p, HKU64 idx);           \
    HK_EXTERN_C HK_DLL_FUNCTION HK##TYPE*             HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getValue)(HKArray##TYPE *p, HKU64 idx);                       \
    HK_EXTERN_C HK_DLL_FUNCTION const HK##TYPE*const* HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getPointer_const)(const HKArray##TYPE *p);                    \
    HK_EXTERN_C HK_DLL_FUNCTION HK##TYPE**            HK_DLL_FUNCTION_NAME(HKArray##TYPE##_internal_getPointer)(HKArray##TYPE *p)
    

#if !defined(HK_RUNTIME_LOAD)
#define HK_OBJECT_ARRAY_DEFINE_COMMON_INLINE(TYPE)                                                                                                                \
    HK_INLINE HKBool HKArray##TYPE##_isEmpty(const HKArray##TYPE *p) { return HKArray##TYPE##_getCount(p) == 0; }                                                 \
    HK_INLINE void   HKArray##TYPE##_clear(HKArray##TYPE *p) { HKArray##TYPE##_setCount(p, 0); }
#else
#define HK_OBJECT_ARRAY_DEFINE_COMMON_INLINE(TYPE)
#endif

#define HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE)       \
    HK_OBJECT_ARRAY_DEFINE_COMMON_DLL(TYPE);      \
    HK_OBJECT_ARRAY_DEFINE_COMMON_INLINE(TYPE);   \
    HK_OBJECT_C_DERIVE_METHODS(HKArray##TYPE);    \
    HK_NAMESPACE_TYPE_ALIAS(Array##TYPE)

#define HK_OBJECT_ARRAY_METHOD_DEFINE(TYPE)                                                           \
    static inline HK_CXX_CONSTEXPR HKUUID TypeID() { return HK_OBJECT_TYPEID_Array##TYPE; }           \
    typedef HK##TYPE *value_type;                                                                     \
    virtual HKArray##TYPE *HK_API clone() const = 0;                                                  \
    virtual HKU64 HK_API getCount() const = 0;                                                        \
    virtual void HK_API  setCount(HKU64 count) = 0;                                                   \
    virtual HKU64 HK_API getCapacity() const = 0;                                                     \
    virtual void HK_API setCapacity(HKU64 count) = 0;                                                 \
    virtual void HK_API setValue(HKU64 idx, HK##TYPE *v) = 0;                                         \
    virtual const HK##TYPE *HK_API internal_getValue_const(HKU64 idx) const = 0;                      \
    virtual HK##TYPE *HK_API internal_getValue(HKU64 idx) = 0;                                        \
    virtual const HK##TYPE *const *HK_API internal_getPointer_const() const = 0;                      \
    virtual HK##TYPE **HK_API internal_getPointer() = 0;                                              \
    HK_INLINE const HK##TYPE *const *getPointer() const { return internal_getPointer_const(); }       \
    HK_INLINE HK##TYPE **getPointer() { return internal_getPointer(); }                               \
    HK_INLINE HKBool isEmpty() const { return getCount() == 0; }                                      \
    HK_INLINE HKArray##TYPE *cloneWithRef() const                                                     \
    {                                                                                                 \
        HKArray##TYPE *ptr = clone();                                                                 \
        ptr->addRef();                                                                                \
        return ptr;                                                                                   \
    }                                                                                                 \
    HK_INLINE void clear() { setCount(0); }                                                           \
    HK_INLINE void resize(HKU64 count) { setCount(count); }                                           \
    HK_INLINE void reserve(HKU64 count) { setCapacity(count); }

#if defined(__cplusplus)
#define HK_OBJECT_ARRAY_DEFINE(TYPE)                                                      \
    struct HKArray##TYPE : public HKUnknown                                               \
    {                                                                                     \
        HK_OBJECT_ARRAY_METHOD_DEFINE(TYPE);                                              \
    };                                                                                    \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE);                                                  \
    HK_OBJECT_CREATE_TRAITS(HKArray##TYPE);                                               \
    template <>                                                                           \
    struct HKArrayTraits<HK##TYPE>                                                        \
    {                                                                                     \
        typedef HKArray##TYPE type;                                                       \
    }

#else
#define HK_OBJECT_ARRAY_DEFINE(TYPE)            \
    typedef struct HKArray##TYPE HKArray##TYPE; \
    HK_OBJECT_ARRAY_DEFINE_COMMON(TYPE)
#endif

#define HK_OBJECT_ARRAY_IMPL_DEFINE_COMMON(TYPE)                                                                         \
    HK_EXTERN_C HK_DLL HKArray##TYPE *HK_API HKArray##TYPE##_create()                                                    \
    {                                                                                                                    \
        auto res = new HKArray##TYPE##Impl();                                                                            \
        res->addRef();                                                                                                   \
        return res;                                                                                                      \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HKArray##TYPE *HK_API HKArray##TYPE##_clone(const HKArray##TYPE *p)                               \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            HKArray##TYPE *ptr = p->cloneWithRef();                                                                      \
            return ptr;                                                                                                  \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HKArray##TYPE *HK_API HKArray##TYPE##_cloneWithoutRef(const HKArray##TYPE *p)                     \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->clone();                                                                                           \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HKU64 HK_API HKArray##TYPE##_getCount(const HKArray##TYPE *p)                                     \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->getCount();                                                                                        \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return 0;                                                                                                    \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL void HK_API HKArray##TYPE##_setCount(HKArray##TYPE *p, HKU64 c)                                   \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->setCount(c);                                                                                       \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HKU64 HK_API HKArray##TYPE##_getCapacity(const HKArray##TYPE *p)                                  \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->getCapacity();                                                                                     \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return 0;                                                                                                    \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL void HK_API HKArray##TYPE##_setCapacity(HKArray##TYPE *p, HKU64 c)                                \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->setCapacity(c);                                                                                    \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL void HK_API HKArray##TYPE##_setValue(HKArray##TYPE *p, HKU64 idx, HK##TYPE *v)                    \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->setValue(idx, v);                                                                                  \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HK##TYPE *HK_API HKArray##TYPE##_internal_getValue(HKArray##TYPE *p, HKU64 idx)                   \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->internal_getValue(idx);                                                                            \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL const HK##TYPE *HK_API HKArray##TYPE##_internal_getValue_const(const HKArray##TYPE *p, HKU64 idx) \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->internal_getValue_const(idx);                                                                      \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL const HK##TYPE *const *HK_API HKArray##TYPE##_internal_getPointer_const(const HKArray##TYPE *p)   \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->internal_getPointer_const();                                                                       \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }                                                                                                                    \
    HK_EXTERN_C HK_DLL HK##TYPE **HK_API HKArray##TYPE##_internal_getPointer(HKArray##TYPE *p)                           \
    {                                                                                                                    \
        if (p)                                                                                                           \
        {                                                                                                                \
            return p->internal_getPointer();                                                                             \
        }                                                                                                                \
        else                                                                                                             \
        {                                                                                                                \
            return nullptr;                                                                                              \
        }                                                                                                                \
    }

#define HK_OBJECT_ARRAY_METHOD_IMPL_DEFINE(TYPE)                                                             \
    virtual void HK_API destroyObject() override                                                             \
    {                                                                                                        \
        for (auto &elm : arr)                                                                                \
        {                                                                                                    \
            if (elm)                                                                                         \
            {                                                                                                \
                elm->release();                                                                              \
            }                                                                                                \
        }                                                                                                    \
    }                                                                                                        \
    virtual HKArray##TYPE *HK_API clone() const override                                                     \
    {                                                                                                        \
        auto res = new HKArray##TYPE##Impl(arr);                                                             \
        for (auto &elm : arr)                                                                                \
        {                                                                                                    \
            if (elm)                                                                                         \
            {                                                                                                \
                elm->addRef();                                                                               \
            }                                                                                                \
        }                                                                                                    \
        return res;                                                                                          \
    }                                                                                                        \
    virtual HKU64 HK_API getCount() const override { return arr.size(); };                                   \
    virtual void HK_API setCount(HKU64 count) override                                                       \
    {                                                                                                        \
        if (count < arr.size())                                                                              \
        {                                                                                                    \
            for (HKU64 i = count; i < arr.size(); ++i)                                                       \
            {                                                                                                \
                if (arr[i])                                                                                  \
                {                                                                                            \
                    arr[i]->release();                                                                       \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
        arr.resize(count);                                                                                   \
    };                                                                                                       \
    virtual HKU64 HK_API getCapacity() const override { return arr.capacity(); };                            \
    virtual void HK_API setCapacity(HKU64 count) override { return arr.reserve(count); };                    \
    virtual void HK_API setValue(HKU64 idx, HK##TYPE *v) override                                            \
    {                                                                                                        \
        if (idx < arr.size())                                                                                \
        {                                                                                                    \
            if (arr[idx] != v)                                                                               \
            {                                                                                                \
                if (arr[idx])                                                                                \
                {                                                                                            \
                    arr[idx]->release();                                                                     \
                };                                                                                           \
                if (v)                                                                                       \
                {                                                                                            \
                    v->addRef();                                                                             \
                }                                                                                            \
                arr[idx] = v;                                                                                \
            }                                                                                                \
        }                                                                                                    \
    }                                                                                                        \
    virtual const HK##TYPE *HK_API internal_getValue_const(HKU64 idx) const override                         \
    {                                                                                                        \
        if (idx < arr.size())                                                                                \
        {                                                                                                    \
            return arr[idx];                                                                                 \
        }                                                                                                    \
        else                                                                                                 \
        {                                                                                                    \
            return nullptr;                                                                                  \
        }                                                                                                    \
    }                                                                                                        \
    virtual HK##TYPE *HK_API internal_getValue(HKU64 idx) override                                           \
    {                                                                                                        \
        if (idx < arr.size())                                                                                \
        {                                                                                                    \
            return arr[idx];                                                                                 \
        }                                                                                                    \
        else                                                                                                 \
        {                                                                                                    \
            return nullptr;                                                                                  \
        }                                                                                                    \
    }                                                                                                        \
    virtual const HK##TYPE *const *HK_API internal_getPointer_const() const override { return arr.data(); }; \
    virtual HK##TYPE **HK_API internal_getPointer() override { return arr.data(); };                         \
    std::vector<HK##TYPE *> arr;

#define HK_OBJECT_ARRAY_IMPL_DEFINE(TYPE)                                                                          \
    struct HK_DLL HKArray##TYPE##Impl : public HKArray##TYPE, protected HKRefCntObject                             \
    {                                                                                                              \
        HKArray##TYPE##Impl() : HKArray##TYPE(), HKRefCntObject(), arr{} {}                                        \
        HKArray##TYPE##Impl(const std::vector<HK##TYPE *> &arr_) : HKArray##TYPE(), HKRefCntObject(), arr{arr_} {} \
        virtual HK_API ~HKArray##TYPE##Impl() {}                                                                   \
        virtual HKU32 HK_API addRef() override { return HKRefCntObject::addRef(); }                                \
        virtual HKU32 HK_API release() override { return HKRefCntObject::release(); }                              \
        virtual HKBool HK_API queryInterface(HKUUID iid, void **ppvInterface) override                             \
        {                                                                                                          \
            if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Array##TYPE)                            \
            {                                                                                                      \
                addRef();                                                                                          \
                *ppvInterface = this;                                                                              \
                return true;                                                                                       \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                return false;                                                                                      \
            }                                                                                                      \
        }                                                                                                          \
        HK_OBJECT_ARRAY_METHOD_IMPL_DEFINE(TYPE);                                                                  \
    };                                                                                                             \
    HK_OBJECT_ARRAY_IMPL_DEFINE_COMMON(TYPE)

#endif