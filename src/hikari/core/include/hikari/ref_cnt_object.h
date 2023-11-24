#ifndef HK_REF_CNT_OBJECT__H
#define HK_REF_CNT_OBJECT__H
#include <hikari/data_type.h>

#if defined(__cplusplus)
#include <atomic>
struct HK_DLL HKRefCntObject {
	virtual HK_API ~HKRefCntObject() HK_CXX_NOEXCEPT;
	virtual void  HK_API destroyObject() = 0;
	HKU32 HK_API addRef() ;
	HKU32 HK_API release();
private:
	std::atomic_ulong m_ref_cnt{ 0 };
#ifndef NDEBUG
	std::atomic_uint32_t debug_idx{ 0 };
#endif
};
namespace hk { typedef ::HKRefCntObject RefCntObject; }
#endif

#endif
