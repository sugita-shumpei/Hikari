#define  HK_DLL_EXPORT 
#include "ref_cnt_object.h"
// #include <cstdio>
HKU32 HK_API HKRefCntObject::addRef()
{
    auto cnt = m_ref_cnt.fetch_add(+1)+1;
    // printf(" addRef=%d->%d\n", cnt-1, cnt);
    return cnt;
}

HKU32 HK_API HKRefCntObject::release()
{
    auto cnt = m_ref_cnt.fetch_sub(1) - 1;
    // printf("release=%d->%d\n", cnt+1, cnt);
    return cnt;
}
