#ifndef HK_REF_PTR__H
#define HK_REF_PTR__H

#include "object.h"
#if defined(__cplusplus)
template<typename T>
struct HKRefPtr {
    HKRefPtr() HK_CXX_NOEXCEPT : m_ptr{nullptr}{}
    HKRefPtr(T* ptr) HK_CXX_NOEXCEPT : m_ptr{ ptr } { HKObject_addRef(ptr); }
    // CONSTRUCTOR
    HKRefPtr(const HKRefPtr<T>& ref) HK_CXX_NOEXCEPT : m_ptr{ref.m_ptr} {
        HKObject_addRef(ref.m_ptr);
    }
    HKRefPtr(HKRefPtr<T>&& ref) HK_CXX_NOEXCEPT  : m_ptr{ref.m_ptr}{
        ref.m_ptr = nullptr;
    }
    // COPY AND MOVE
    HKRefPtr& operator=(const HKRefPtr<T>& ref) HK_CXX_NOEXCEPT{
        if (this!=&ref){
            // 新しいポインタの参照を増やす
            HKObject_addRef(ref.m_ptr);
            // 古いポインタの参照を減らす
            HKObject_release(m_ptr);
            // 入れ替える
            m_ptr = ref.m_ptr;
        }
        return *this;
    }
    HKRefPtr& operator=(HKRefPtr<T>&& ref) HK_CXX_NOEXCEPT{
        if (this!=&ref){
            // 古いポインタの参照を減らす
            HKObject_release(m_ptr);
            // 入れ替える
            m_ptr = ref.m_ptr;
            ref.m_ptr = nullptr;
        }
        return *this;
    }
    // CAST
    template<typename U>
    HKRefPtr(const HKRefPtr<U>& ref) HK_CXX_NOEXCEPT{
        // ポインターキャスト
        HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
    }
    template<typename U>
    HKRefPtr(HKRefPtr<U>&& ref) HK_CXX_NOEXCEPT{
        // 新しいポインタの参照を増やす
        HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
        // 新しいポインタの参照を減らす
        HKObject_release(m_ptr);
        ref.m_ptr = nullptr;
    }
    template<typename U>
    HKRefPtr& operator=(const HKRefPtr<U>& ref) HK_CXX_NOEXCEPT{
        if (this != &ref){
            T* ptr = m_ptr;
            // ポインターキャスト
            HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
            //  古いポインタの参照を減らす
            HKObject_release(ptr);
        }
        return *this;
    }
    template<typename U>
    HKRefPtr& operator=(HKRefPtr<U>&& ref) HK_CXX_NOEXCEPT{
        if (this != &ref){
            // 新しいポインタの参照を増やす
            HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
            HKObject_release(ref.m_ptr);
            // 古いポインタの参照を減らす
            HKObject_release(m_ptr);
            ref.m_ptr = nullptr;
        }
        return *this;
    }
    // Destructor
    ~HKRefPtr() HK_CXX_NOEXCEPT {
        reset(nullptr);
    }
    // 比較演算子
    HKBool operator==(const HKRefPtr<T>& ref) const HK_CXX_NOEXCEPT {
        return ref.m_ptr == m_ptr;
    }
    HKBool operator!=(const HKRefPtr<T>& ref) const HK_CXX_NOEXCEPT {
        return ref.m_ptr != m_ptr;
    }
    // ALLOW OPERATOR
    T* operator->() HK_CXX_NOEXCEPT { return m_ptr; }
    const T* operator->() const HK_CXX_NOEXCEPT { return m_ptr;}
    // GET
    T* get() HK_CXX_NOEXCEPT { return m_ptr; }
    const T* get() const HK_CXX_NOEXCEPT { return m_ptr; }
    // GET ADDRESS OF
    T** getAddressOf()HK_CXX_NOEXCEPT {
        return &m_ptr;
    }
    const T** getAddressOf() const HK_CXX_NOEXCEPT {
        return &m_ptr;
    }
    // reset
    void reset(T* ptr = nullptr, bool incRef = true) {
        HKObject_release(m_ptr);
        if (incRef) {
            HKObject_addRef(ptr);
        }
        m_ptr = ptr;
    }
    // cast
    template<typename T>
    HKBool queryInterface(HKRefPtr<T>& ptr) {
        T* tmp = nullptr;
        if (HKObject_queryInterface(m_ptr, T::TypeID(), (void**)&tmp)) {
            ptr.reset(tmp,false);
            
            return true;
        }
        return false;
    }

    template<typename ...Args>
    static HKRefPtr<T> create(Args&& ...args) { 
        return HKRefPtr<T>(T::create(args...), nullptr);
    }
private:
    HKRefPtr(T* ptr, void* dmy) HK_CXX_NOEXCEPT : m_ptr{ ptr } { }
private:
    T* m_ptr;
};
#endif

#endif
