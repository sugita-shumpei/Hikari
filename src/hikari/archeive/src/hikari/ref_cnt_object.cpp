#define  HK_DLL_EXPORT 
#include <hikari/ref_cnt_object.h>
#ifndef NDEBUG
#include <cstdio>
#include <cassert>
#endif

static HKU32 HKRefCnt_internal_debugIdx() {
    static std::atomic<uint32_t> cnt;
    return cnt.fetch_add(+1) + 1;
}

HK_API HKRefCntObject::~HKRefCntObject() HK_CXX_NOEXCEPT {
    assert(m_ref_cnt.load() == 0);
}
// #include <cstdio>
HKU32 HK_API HKRefCntObject::addRef()
{
    auto cnt = m_ref_cnt.fetch_add(+1)+1;
#ifndef NDEBUG
    HKU32 val = 0;
    debug_idx.compare_exchange_strong(val, HKRefCnt_internal_debugIdx());
    printf("%d: addRef=%d->%d\n", debug_idx.load(), cnt - 1, cnt);
#endif
    return cnt;
}

HKU32 HK_API HKRefCntObject::release()
{
    auto cnt = m_ref_cnt.fetch_sub(1);
    assert(cnt >= 1);
#ifndef NDEBUG
    printf("%d: release=%d->%d\n", debug_idx.load(), cnt, cnt - 1);
#endif
    if (cnt == 1) {
        destroyObject();
        delete this;
    }

    return cnt - 1;
}
