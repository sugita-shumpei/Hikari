#define  HK_DLL_EXPORT 
#include "ref_cnt_object.h"
#ifndef NDEBUG
#include <cstdio>
#include <cassert>
#endif
// #include <cstdio>
HKU32 HK_API HKRefCntObject::addRef()
{
    auto cnt = m_ref_cnt.fetch_add(+1)+1;
#ifndef NDEBUG
    printf(" addRef=%d->%d\n", cnt-1, cnt);
#endif
    return cnt;
}

HKU32 HK_API HKRefCntObject::release()
{
    auto cnt = m_ref_cnt.fetch_sub(1);
    assert(cnt >= 1);
    if (cnt == 1) {
        destroyObject();
        delete this;
    }
#ifndef NDEBUG
    printf("release=%d->%d\n", cnt, cnt-1);
#endif
    return cnt - 1;
}
