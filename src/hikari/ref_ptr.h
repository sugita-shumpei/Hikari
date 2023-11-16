#ifndef HK_REF_PTR__H
#define HK_REF_PTR__H

#include "object.h"

#if defined(__cplusplus)
template<typename T>
struct HKRefPtr {
    HKRefPtr() HK_CXX_NOEXCEPT : m_ptr{nullptr}{}
    HKRefPtr(T* ptr) HK_CXX_NOEXCEPT : m_ptr{ ptr } {}
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
        return HKRefPtr<T>(T::create(args...));
    }
private:
    T* m_ptr;
};

template<typename T>
HK_INLINE HKRefPtr<T> HKRefPtr_makeRef(T* ptr) {
    if (!ptr) { return nullptr; }
    else {
        ptr->addRef();
        return HKRefPtr<T>(ptr);
    }
}

template<typename T>
struct HKArrayRefPtr {
    typedef typename T::value_type value_type; 
    HKArrayRefPtr() HK_CXX_NOEXCEPT : m_ptr{nullptr}{}
    HKArrayRefPtr(T* ptr) HK_CXX_NOEXCEPT : m_ptr{ ptr } {}
    // CONSTRUCTOR
    HKArrayRefPtr(const HKArrayRefPtr<T>& ref) HK_CXX_NOEXCEPT : m_ptr{ref.m_ptr} {
        HKObject_addRef(ref.m_ptr);
    }
    HKArrayRefPtr(HKArrayRefPtr<T>&& ref) HK_CXX_NOEXCEPT  : m_ptr{ref.m_ptr}{
        ref.m_ptr = nullptr;
    }
    // COPY AND MOVE
    HKArrayRefPtr& operator=(const HKArrayRefPtr<T>& ref) HK_CXX_NOEXCEPT{
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
    HKArrayRefPtr& operator=(HKArrayRefPtr<T>&& ref) HK_CXX_NOEXCEPT{
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
    HKArrayRefPtr(const HKArrayRefPtr<U>& ref) HK_CXX_NOEXCEPT{
        // ポインターキャスト
        HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
    }
    template<typename U>
    HKArrayRefPtr(HKArrayRefPtr<U>&& ref) HK_CXX_NOEXCEPT{
        // 新しいポインタの参照を増やす
        HKObject_queryInterface(ref.m_ptr,T::TypeID(),(void**)m_ptr);
        // 新しいポインタの参照を減らす
        HKObject_release(m_ptr);
        ref.m_ptr = nullptr;
    }
    template<typename U>
    HKArrayRefPtr& operator=(const HKArrayRefPtr<U>& ref) HK_CXX_NOEXCEPT{
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
    HKArrayRefPtr& operator=(HKArrayRefPtr<U>&& ref) HK_CXX_NOEXCEPT{
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
    ~HKArrayRefPtr() HK_CXX_NOEXCEPT {
        reset(nullptr);
    }
    // 
    bool isEmpty() const { return size() == 0; }
    // リサイズ
    void  resize(HKU64 new_size) { if (m_ptr) { m_ptr->resize(new_size); } }
    // リザーブ
    void  reserve(HKU64 new_size) { if (m_ptr) { m_ptr->reserve(new_size); } }
    // サイズ
    HKU64 size() const { if (m_ptr) { return m_ptr->getCount(); } else { return 0; } }
    HKU64 capacity() const { if (m_ptr) { return m_ptr->getCapacity(); } else { return 0; } }
    // アクセッサ
    value_type& operator[](HKU64 idx) {
        value_type* ptr = (m_ptr->internal_getPointer() + idx);
        return *ptr;
    }
    const value_type& operator[](HKU64 idx) const {
        const value_type* ptr = (m_ptr->internal_getPointer_const() + idx);
        return *ptr;
    }
    // クリア
    void clear() { if (m_ptr) { m_ptr->clear(); } }
    // 比較演算子
    HKBool operator==(const HKArrayRefPtr<T>& ref) const HK_CXX_NOEXCEPT {
        return ref.m_ptr == m_ptr;
    }
    HKBool operator!=(const HKArrayRefPtr<T>& ref) const HK_CXX_NOEXCEPT {
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
    HKBool queryInterface(HKArrayRefPtr<T>& ptr) {
        T* tmp = nullptr;
        if (HKObject_queryInterface(m_ptr, T::TypeID(), (void**)&tmp)) {
            ptr.reset(tmp,false);
            
            return true;
        }
        return false;
    }

    template<typename ...Args>
    static HKArrayRefPtr<T> create(Args&& ...args) { 
        return HKArrayRefPtr<T>(T::create(args...));
    }
private:
    T* m_ptr;
};

template<typename T>
HK_INLINE HKArrayRefPtr<T> HKArrayRefPtr_makeRef(T* ptr) {
    if (!ptr) { return nullptr; }
    else {
        ptr->addRef();
        return HKArrayRefPtr<T>(ptr);
    }
}

#endif

#endif
