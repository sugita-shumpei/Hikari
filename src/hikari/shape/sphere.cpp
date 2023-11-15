#define HK_DLL_EXPORT
#include "sphere.h"
#include "../ref_cnt_object.h"
#include <shared_mutex>

struct HK_DLL HKSphereImpl : public HKSphere, protected HKRefCntObject {
    // HKRefCntObject ����Čp������܂���
    HKSphereImpl() ;
    HKSphereImpl(const HKVec3& center_, HKF32 radius_);
    virtual ~HKSphereImpl();
    virtual HKU32  HK_API addRef() override  { return HKRefCntObject::addRef(); }
    virtual HKU32  HK_API release() override { return HKRefCntObject::release(); }
    virtual HKBool HK_API queryInterface(HKUUID iid, void** ppvInterface) override
    {
        if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_Shape || iid == HK_OBJECT_TYPEID_Sphere) {
            addRef();
            *ppvInterface = this;
            return true;
        }
        return false;
    }
    virtual void   HK_API setCenter(const HKVec3& c) override
    {
        std::lock_guard<std::shared_mutex> lk{ m_mutex };
        m_center = c;
    }
    virtual HKVec3 HK_API getCenter() const  override
    {
        std::shared_lock<std::shared_mutex> lk{ m_mutex };
        return m_center;
    }
    virtual void   HK_API setRadius(HKF32 r) override
    {
        std::lock_guard<std::shared_mutex> lk{ m_mutex };
        m_radius = r;
    }
    virtual HKF32  HK_API getRadius() const  override
    {
        std::shared_lock<std::shared_mutex> lk{ m_mutex };
        return m_radius;
    }
    virtual HKAabb HK_API getAabb() const override {
        std::shared_lock<std::shared_mutex> lk{ m_mutex };
        return HKAabb(m_center - HKVec3(m_radius * 0.5f), m_center + HKVec3(m_radius * 0.5f));
    }
private:
    void HK_API destroyObject() override {}
private:
    mutable std::shared_mutex m_mutex;
    HKVec3 m_center;
    HKF32  m_radius;
};


HKSphereImpl::HKSphereImpl(): HKSphere(),HKRefCntObject(), m_center{0.0f,0.0f,0.0f}, m_radius{1.0f}
{
}

HKSphereImpl::HKSphereImpl(const HKVec3& center_, HKF32 radius_) : HKSphere(), HKRefCntObject(),m_center{ center_ }, m_radius{ radius_ }
{

}

HKSphereImpl::~HKSphereImpl()
{
}

HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create()
{
    auto sphere = new HKSphereImpl();
    sphere->addRef();
    return sphere;
}

HK_EXTERN_C HK_DLL HKSphere* HK_API HKSphere_create2(HKCVec3 c, HKF32 r)
{
    auto sphere = new HKSphereImpl(c,r);
    sphere->addRef();
    return sphere;
}

HK_EXTERN_C HK_DLL void HK_API HKSphere_setCenter(HKSphere* sp, HKCVec3 c)
{
    if (sp) {
        return sp->setCenter(c);
    }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKSphere_getCenter(const HKSphere* sp)
{
    if (sp) {
        return sp->getCenter();
    }
    else {
        return HKCVec3();
    }
}

HK_EXTERN_C HK_DLL void HK_API HKSphere_setRadius(HKSphere* sp, HKF32 r)
{
    if (sp) {
        return sp->setRadius(r);
    }
}

HK_EXTERN_C HK_DLL HKF32 HK_API HKSphere_getRadius(const HKSphere* sp)
{
    if (sp) {
        return sp->getRadius();
    }
    else {
        return 0.0f;
    }
}
