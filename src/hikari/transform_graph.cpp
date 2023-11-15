#define HK_DLL_EXPORT
#include "transform_graph.h"
#include <mutex>
#include <vector>

struct HK_DLL HKTransformGraphNodeImpl : public HKTransformGraphNode, protected HKRefCntObject {
    typedef HKTransformGraphNode Node;

    HKTransformGraphNodeImpl();
    virtual            HK_API ~HKTransformGraphNodeImpl();
    virtual HKU32      HK_API addRef() override;
    virtual HKU32      HK_API release() override;
    virtual HKBool     HK_API queryInterface(HKUUID iid, void** ppvInterface) override;
    virtual void       HK_API setParent(HKTransformGraphNode* parent) override;
    virtual HKU32      HK_API getChildCount() const override;
    virtual void       HK_API setObject(HKUnknown* object) override;

    virtual HKAffine3D HK_API getLocalTransform() const override;
    virtual void       HK_API setLocalTransform(const HKAffine3D& transform) override;
    virtual HKQuat     HK_API getLocalRotation() const override;
    virtual HKVec3     HK_API getLocalPosition() const override;
    virtual HKVec3     HK_API getLocalScaling()  const override;
    virtual void       HK_API setLocalRotation(const HKQuat& rotation) override;
    virtual void       HK_API setLocalPosition(const HKVec3& position) override;
    virtual void       HK_API setLocalScaling(const HKVec3&   scaling) override;

    virtual HKVec3     HK_API getPosition() const override;
    virtual void       HK_API setPosition(const HKVec3& position) override;
    virtual HKMat4x4   HK_API getTransformPointMatrix() const override;
    virtual HKMat4x4   HK_API getTransformVectorMatrix() const override;

    virtual HKVec3     HK_API transformPoint(const HKVec3& p) const override;
    virtual HKVec3     HK_API transformVector(const HKVec3& v) const override;
    virtual HKVec3     HK_API transformDirection(const HKVec3& d) const override;
    virtual HKVec3     HK_API inverseTransformPoint(const HKVec3& p) const override;
    virtual HKVec3     HK_API inverseTransformVector(const HKVec3& v) const override;
    virtual HKVec3     HK_API inverseTransformDirection(const HKVec3& d) const override;

    virtual HKUnknown*       HK_API internal_getObject() override;
    virtual const HKUnknown* HK_API internal_getObject_const() const override;
    virtual Node*            HK_API internal_getParent() override;
    virtual const Node*      HK_API internal_getParent_const() const override;
    virtual Node*            HK_API internal_getChild(HKU32 idx) override;
    virtual const Node*      HK_API internal_getChild_const(HKU32 idx) const override;

private:
    virtual void HK_API destroyObject() override;
    void HK_API internal_popChild(Node* child);
    void HK_API internal_addChild(Node* child);
    void HK_API internal_update();
    void HK_API internal_updateParent(const HKMat4x4& new_parent_transform_point_matrix, const HKMat4x4& new_parent_transform_vector_matrix);
    HK_INLINE std::vector<HKTransformGraphNode*> internal_getChildrenRecursive()
    {
        std::unique_lock<std::mutex> lk{ m_mutex_children };
        auto res = std::vector<HKTransformGraphNode*>();
        for (auto child: m_children) {
            auto tmp = ((HKTransformGraphNodeImpl*)child)->internal_getChildrenRecursive();
            res.reserve(tmp.size() + res.size());
            for (auto& t : tmp) { res.push_back(t); }
        }
        return res;
    }
    // �s����X�V����
    inline void internal_mulMatrix(const HKMat4x4& rel_point_matrix, const HKMat4x4& rel_vector_matrix) {
        std::unique_lock<std::mutex> lk{ m_mutex_transform };
        m_parent_transform_point_matrix  = rel_point_matrix  * m_parent_transform_point_matrix;
        m_parent_transform_vector_matrix = rel_vector_matrix * m_parent_transform_vector_matrix;
    }
private:
    mutable std::mutex                 m_mutex_parent;
    mutable std::mutex                 m_mutex_children;
    mutable std::mutex                 m_mutex_object;
    mutable std::mutex                 m_mutex_transform;
    HKMat4x4                           m_parent_transform_point_matrix;
    HKMat4x4                           m_parent_transform_vector_matrix;
    HKAffine3D                         m_local_transform;
    HKUnknown*                         m_object;
    HKTransformGraphNode*              m_parent;
    std::vector<HKTransformGraphNode*> m_children;
};

struct HKTransformGraphImpl : public HKTransformGraph, protected HKRefCntObject {
    typedef HKTransformGraphNode Node;

    HKTransformGraphImpl();
    virtual            HK_API ~HKTransformGraphImpl();
    virtual HKU32      HK_API addRef() override;
    virtual HKU32      HK_API release() override;
    virtual HKBool     HK_API queryInterface(HKUUID iid, void** ppvInterface) override;
    virtual void       HK_API setChild(HKTransformGraphNode* rootNode) override;
    virtual HKU32      HK_API getChildCount() const override;
    virtual void       HK_API setObject(HKUnknown* object) override;

    virtual HKUnknown*       HK_API internal_getObject() override;
    virtual const HKUnknown* HK_API internal_getObject_const() const override;
    virtual const Node*      HK_API internal_getRootNode_const() const override;
    virtual Node*            HK_API internal_getChild(HKU32 idx)override;
    virtual const Node*      HK_API internal_getChild_const(HKU32 idx) const override;

    virtual HKAffine3D HK_API getLocalTransform() const override;
    virtual void       HK_API setLocalTransform(const HKAffine3D& transform) override;
    virtual HKQuat     HK_API getLocalRotation() const override;
    virtual HKVec3     HK_API getLocalPosition() const override;
    virtual HKVec3     HK_API getLocalScaling()  const override;
    virtual void       HK_API setLocalRotation(const HKQuat& rotation) override;
    virtual void       HK_API setLocalPosition(const HKVec3& position) override;
    virtual void       HK_API setLocalScaling(const HKVec3& scaling) override;
    virtual HKVec3     HK_API getPosition() const override;
    virtual void       HK_API setPosition(const HKVec3& position) override;
    virtual HKMat4x4   HK_API getTransformPointMatrix() const override;
    virtual HKMat4x4   HK_API getTransformVectorMatrix() const override;
    virtual HKVec3     HK_API transformPoint(const HKVec3& p) const override;
    virtual HKVec3     HK_API transformVector(const HKVec3& v) const override;
    virtual HKVec3     HK_API transformDirection(const HKVec3& d) const override;
    virtual HKVec3     HK_API inverseTransformPoint(const HKVec3& p) const override;
    virtual HKVec3     HK_API inverseTransformVector(const HKVec3& v) const override;
    virtual HKVec3     HK_API inverseTransformDirection(const HKVec3& d) const override;

private:
    virtual void HK_API destroyObject() override;
private:
    mutable std::mutex    m_mutex;
    HKTransformGraphNode* m_root_node;
};



HK_EXTERN_C HK_DLL HKTransformGraphNode*  HK_API HKTransformGraphNode_create()
{
    auto res = new HKTransformGraphNodeImpl();
    res->addRef();
    return res;
}

HK_EXTERN_C HK_DLL void  HK_API HKTransformGraphNode_setParent(HKTransformGraphNode* node, HKTransformGraphNode* parent)
{
    // �܂��e�����݂��邩�ǂ���
    if (node) {
        node->setParent(parent);
    }
}

HK_EXTERN_C HK_DLL HKU32  HK_API HKTransformGraphNode_getChildCount(const HKTransformGraphNode* node)
{
    if (node) { return node->getChildCount(); }
    else { return 0; }
}

HK_EXTERN_C HK_DLL void  HK_API HKTransformGraphNode_setObject(HKTransformGraphNode* node, HKUnknown* object)
{
    if (node) { return node->setObject(object); }
}

HK_EXTERN_C HK_DLL HKCAffine3D  HK_API HKTransformGraphNode_getLocalTransform(const HKTransformGraphNode* node)
{
    if (node) { return node->getLocalTransform(); }
    else { return {};  }
}

HK_EXTERN_C HK_DLL HKCQuat HK_API HKTransformGraphNode_getLocalRotation(const HKTransformGraphNode* node)
{
    if (node) { return node->getLocalRotation(); }
    else { return HKQuat(1.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_getLocalPosition(const HKTransformGraphNode* node)
{
    if (node) { return node->getLocalPosition(); }
    else { return HKVec3(0.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_getLocalScaling(const HKTransformGraphNode* node)
{
    if (node) { return node->getLocalScaling(); }
    else { return HKVec3(1.0f); }
}

HK_EXTERN_C HK_DLL void  HK_API HKTransformGraphNode_setLocalTransform(HKTransformGraphNode* node, HKCAffine3D transform)
{
    if (node) { return node->setLocalTransform(transform); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraphNode_setLocalRotation(HKTransformGraphNode* node, HKCQuat rotation)
{
    if (node) { return node->setLocalRotation(rotation); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraphNode_setLocalPosition(HKTransformGraphNode* node, HKCVec3 position)
{
    if (node) { return node->setLocalPosition(position); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraphNode_setLocalScaling(HKTransformGraphNode* node, HKCVec3 scaling)
{
    if (node) { return node->setLocalScaling(scaling); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_getPosition(const HKTransformGraphNode* node)
{
    if (node) { return node->getPosition(); }
    else { return HKVec3(0.0f); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraphNode_setPosition(HKTransformGraphNode* node, HKCVec3 position)
{
    if (node) { return node->setPosition(position); }
}

HK_EXTERN_C HK_DLL HKCMat4x4 HK_API HKTransformGraphNode_getTransformPointMatrix(const HKTransformGraphNode* node)
{
    if (node) { return node->getTransformPointMatrix(); }
    else { return HKMat4x4(1.0f); }
}

HK_EXTERN_C HK_DLL HKCMat4x4 HK_API HKTransformGraphNode_getTransformVectorMatrix(const HKTransformGraphNode* node)
{
    if (node) { return node->getTransformVectorMatrix(); }
    else { return HKMat4x4(1.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_transformPoint(HKTransformGraphNode* node, HKCVec3 p)
{
    if (node) { return node->transformPoint(p); }
    else { return p; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_transformVector(HKTransformGraphNode* node, HKCVec3 v)
{
    if (node) { return node->transformVector(v); }
    else { return v; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_transformDirection(HKTransformGraphNode* node, HKCVec3 d)
{
    if (node) { return node->transformDirection(d); }
    else { return d; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_inverseTransformPoint(HKTransformGraphNode* node, HKCVec3 p)
{
    if (node) { return node->transformDirection(p); }
    else { return p; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_inverseTransformVector(HKTransformGraphNode* node, HKCVec3 v)
{
    if (node) { return node->transformDirection(v); }
    else { return v; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraphNode_inverseTransformDirection(HKTransformGraphNode* node, HKCVec3 d)
{
    if (node) { return node->transformDirection(d); }
    else { return d; }
}

HK_EXTERN_C HK_DLL HKUnknown*  HK_API HKTransformGraphNode_getObject(HKTransformGraphNode* node)
{
    if (node) { return node->getObject(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKUnknown*  HK_API HKTransformGraphNode_getObject_const(const HKTransformGraphNode* node)
{
    if (node) { return node->internal_getObject_const(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL HKTransformGraphNode*  HK_API HKTransformGraphNode_getParent(HKTransformGraphNode* node)
{
    if (node) { return node->internal_getParent(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKTransformGraphNode*  HK_API HKTransformGraphNode_getParent_const(const HKTransformGraphNode* node)
{
    if (node) { return node->internal_getParent_const(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL HKTransformGraphNode*  HK_API HKTransformGraphNode_getChild(HKTransformGraphNode* node, HKU32 idx)
{
    if (node) { return node->internal_getChild(idx); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKTransformGraphNode*  HK_API HKTransformGraphNode_getChild_const(const HKTransformGraphNode* node, HKU32 idx)
{
    if (node) { return node->internal_getChild_const(idx); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL HKTransformGraph*  HK_API HKTransformGraph_create()
{
    auto res= new HKTransformGraphImpl();
    res->addRef();
    return res;
}

HK_EXTERN_C HK_DLL HKU32  HK_API HKTransformGraph_getChildCount(const HKTransformGraph* root)
{
    if (root) { return root->getChildCount(); }
    else { return 0; }
}

HK_EXTERN_C HK_DLL void  HK_API HKTransformGraph_setObject(HKTransformGraph* root, HKUnknown* object)
{
    if (root) { return root->setObject(object); }
}

HK_EXTERN_C HK_DLL HKCAffine3D  HK_API HKTransformGraph_getLocalTransform(const HKTransformGraph* root)
{
    if (root) { return root->getLocalTransform(); }
    else { return {}; }
}

HK_EXTERN_C HK_DLL HKCQuat HK_API HKTransformGraph_getLocalRotation(const HKTransformGraph* root)
{
    if (root) { return root->getLocalRotation(); }
    else { return HKQuat(1.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_getLocalPosition(const HKTransformGraph* root)
{
    if (root) { return root->getLocalPosition(); }
    else { return HKVec3(0.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_getLocalScaling(const HKTransformGraph* root)
{
    if (root) { return root->getLocalScaling(); }
    else { return HKVec3(1.0f); }
}

HK_EXTERN_C HK_DLL void  HK_API HKTransformGraph_setLocalTransform(HKTransformGraph* root, HKCAffine3D transform)
{
    if (root) { return root->setLocalTransform(transform); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraph_setLocalRotation(HKTransformGraph* root, HKCQuat rotation)
{
    if (root) { return root->setLocalRotation(rotation); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraph_setLocalPosition(HKTransformGraph* root, HKCVec3 position)
{
    if (root) { return root->setLocalPosition(position); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraph_setLocalScaling(HKTransformGraph* root, HKCVec3 scaling)
{
    if (root) { return root->setLocalScaling(scaling); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_getPosition(const HKTransformGraph* root)
{
    if (root) { return root->getPosition(); }
    else { return HKVec3(0.0f); }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraph_setPosition(HKTransformGraph* root, HKCVec3 position)
{
    if (root) { return root->setPosition(position); }
}

HK_EXTERN_C HK_DLL HKCMat4x4 HK_API HKTransformGraph_getTransformPointMatrix(const HKTransformGraph* root)
{
    if (root) { return root->getTransformPointMatrix(); }
    else { return HKMat4x4(1.0f); }
}

HK_EXTERN_C HK_DLL HKCMat4x4 HK_API HKTransformGraph_getTransformVectorMatrix(const HKTransformGraph* root)
{
    if (root) { return root->getTransformVectorMatrix(); }
    else { return HKMat4x4(1.0f); }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_transformPoint(HKTransformGraph* root, HKCVec3 p)
{
    if (root) { return root->transformPoint(p); }
    else { return p; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_transformVector(HKTransformGraph* root, HKCVec3 v)
{
    if (root) { return root->transformVector(v); }
    else { return v; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_transformDirection(HKTransformGraph* root, HKCVec3 d)
{
    if (root) { return root->transformDirection(d); }
    else { return d; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_inverseTransformPoint(HKTransformGraph* root, HKCVec3 p)
{
    if (root) { return root->inverseTransformPoint(p); }
    else { return p; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_inverseTransformVector(HKTransformGraph* root, HKCVec3 v)
{
    if (root) { return root->inverseTransformVector(v); }
    else { return v; }
}

HK_EXTERN_C HK_DLL HKCVec3 HK_API HKTransformGraph_inverseTransformDirection(HKTransformGraph* root, HKCVec3 d)
{
    if (root) { return root->inverseTransformDirection(d); }
    else { return d; }
}

HK_EXTERN_C HK_DLL HKUnknown*  HK_API HKTransformGraph_getObject(HKTransformGraph* root)
{
    if (root) { return root->internal_getObject(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKUnknown*  HK_API HKTransformGraph_getObject_const(const HKTransformGraph* root)
{
    if (root) { return root->internal_getObject_const(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKTransformGraphNode*  HK_API HKTransformGraph_getRootNode_const(const HKTransformGraph* root)
{
    if (root) { return root->internal_getRootNode_const(); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL void HK_API HKTransformGraph_setChild(HKTransformGraph* root, HKTransformGraphNode* node)
{
    if (root) { return root->setChild(node); }
}

HK_EXTERN_C HK_DLL HKTransformGraphNode*  HK_API HKTransformGraph_getChild(HKTransformGraph* root, HKU32 idx)
{
    if (root) { return root->internal_getChild(idx); }
    else { return nullptr; }
}

HK_EXTERN_C HK_DLL const HKTransformGraphNode*  HK_API HKTransformGraph_getChild_const(const HKTransformGraph* root, HKU32 idx)
{
    if (root) { return root->internal_getChild_const(idx); }
    else { return nullptr; }
}

HKTransformGraphNodeImpl::HKTransformGraphNodeImpl()
    :
    m_mutex_children{},
    m_mutex_parent{},
    m_mutex_object{},
    m_mutex_transform{},
    m_parent{nullptr},
    m_parent_transform_point_matrix{1.0f},
    m_parent_transform_vector_matrix{ 1.0f },
    m_children{ },
    m_object{ nullptr },
    m_local_transform{}
{}

HK_API HKTransformGraphNodeImpl::~HKTransformGraphNodeImpl()
{
    if (!m_children.empty()) {
        for (auto& child : m_children) {
            child->release();
        }
    }
}

HKU32 HK_API HKTransformGraphNodeImpl::addRef()
{
    return HKRefCntObject::addRef();
}

HKU32 HK_API HKTransformGraphNodeImpl::release()
{
    return HKRefCntObject::release();
}

HKBool HK_API HKTransformGraphNodeImpl::queryInterface(HKUUID iid, void** ppvInterface)
{
    if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_TransformGraphNode) {
        addRef();
        *ppvInterface = this;
        return true;
    }
    return false;
}

void HK_API HKTransformGraphNodeImpl::setParent(HKTransformGraphNode* parent)
{
    if (parent == m_parent) { return; }
    std::unique_lock<std::mutex> lk{ m_mutex_parent };
    // ���\�[�X�������Ȃ��悤�ɎQ�ƃJ�E���g�𑝂₵�Ă���
    addRef();
    if (m_parent) {
        ((HKTransformGraphNodeImpl*)m_parent)->internal_popChild(this);
        m_parent = nullptr;
    }
    if (parent)   {
        ((HKTransformGraphNodeImpl*)parent)->internal_addChild(this);
        m_parent = parent;
        internal_updateParent(parent->getTransformPointMatrix(), parent->getTransformVectorMatrix());
    }
    else {
        m_parent = nullptr;
        internal_updateParent(HKMat4x4(1.0f), HKMat4x4(1.0f));
    }
    // ���\�[�X�������Ȃ��̂ŎQ�ƃJ�E���g�����炷
    release();
}

HKU32 HK_API HKTransformGraphNodeImpl::getChildCount() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    return m_children.size();
}

void HK_API HKTransformGraphNodeImpl::setObject(HKUnknown* object)
{
    std::unique_lock<std::mutex> lk{ m_mutex_object };
    m_object = object;
}

HKAffine3D HK_API HKTransformGraphNodeImpl::getLocalTransform() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_local_transform;
}

void HK_API HKTransformGraphNodeImpl::setLocalTransform(const HKAffine3D& transform)
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    m_local_transform = transform;
    internal_update();
}

HKQuat HK_API HKTransformGraphNodeImpl::getLocalRotation() const {
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_local_transform.rotation;
}
HKVec3 HK_API HKTransformGraphNodeImpl::getLocalPosition() const {
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_local_transform.position;
}
HKVec3 HK_API HKTransformGraphNodeImpl::getLocalScaling()  const {
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_local_transform.scaling;
}

void HK_API HKTransformGraphNodeImpl::setLocalRotation(const HKQuat& rotation)
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    m_local_transform.rotation = rotation;
    internal_update();

}

void HK_API HKTransformGraphNodeImpl::setLocalPosition(const HKVec3& position)
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    m_local_transform.position = position;
    internal_update();
}

void HK_API HKTransformGraphNodeImpl::setLocalScaling(const HKVec3& scaling)
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    m_local_transform.scaling = scaling;
    internal_update();
}

HKVec3 HK_API HKTransformGraphNodeImpl::getPosition() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto v = (m_parent_transform_point_matrix * HKVec4(m_local_transform.position,1.0f));
    return HKVec3(v.x, v.y, v.z);
}

void HK_API HKTransformGraphNodeImpl::setPosition(const HKVec3& position)
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto v = (m_parent_transform_point_matrix.inverse() * HKVec4(m_local_transform.position, 1.0f));
    m_local_transform.position = HKVec3(v.x, v.y, v.z);
    internal_update();
}

HKMat4x4 HK_API HKTransformGraphNodeImpl::getTransformPointMatrix() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_parent_transform_point_matrix * m_local_transform.transformPointMatrix();
}

HKMat4x4 HK_API HKTransformGraphNodeImpl::getTransformVectorMatrix() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    return m_parent_transform_vector_matrix * m_local_transform.transformVectorMatrix();
}

HKVec3 HK_API HKTransformGraphNodeImpl::transformPoint(const HKVec3& p) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto v = m_parent_transform_point_matrix* HKVec4(m_local_transform.transformPoint(p), 1.0f);
    return HKVec3(v.x, v.y, v.z);
}

HKVec3 HK_API HKTransformGraphNodeImpl::transformVector(const HKVec3& v) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto r = m_parent_transform_vector_matrix * HKVec4(m_local_transform.transformVector(v), 1.0f);
    return HKVec3(r.x, r.y, r.z);
}

HKVec3 HK_API HKTransformGraphNodeImpl::transformDirection(const HKVec3& d) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto r = m_parent_transform_vector_matrix * HKVec4(m_local_transform.transformDirection(d), 1.0f);
    return HKVec3(r.x, r.y, r.z).normalize();
}

HKVec3 HK_API HKTransformGraphNodeImpl::inverseTransformPoint(const HKVec3& p) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto v = m_parent_transform_point_matrix.inverse() * HKVec4(m_local_transform.inverseTransformPoint(p), 1.0f);
    return HKVec3(v.x, v.y, v.z);
}

HKVec3 HK_API HKTransformGraphNodeImpl::inverseTransformVector(const HKVec3& v) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_transform };
    auto r = m_parent_transform_vector_matrix.inverse() * HKVec4(m_local_transform.inverseTransformVector(v), 1.0f);
    return HKVec3(r.x, r.y, r.z);
}

HKVec3 HK_API HKTransformGraphNodeImpl::inverseTransformDirection(const HKVec3& d) const
{
    return (inverseTransformVector(d)).normalize();
}

HKUnknown* HK_API HKTransformGraphNodeImpl::internal_getObject()
{
    std::unique_lock<std::mutex> lk{ m_mutex_object };
    return m_object;
}

const HKUnknown* HK_API HKTransformGraphNodeImpl::internal_getObject_const() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_object };
    return m_object;
}

HKTransformGraphNode* HK_API HKTransformGraphNodeImpl::internal_getParent()
{
    std::unique_lock<std::mutex> lk{ m_mutex_parent };
    return m_parent;
}

const HKTransformGraphNode* HK_API HKTransformGraphNodeImpl::internal_getParent_const() const
{
    std::unique_lock<std::mutex> lk{ m_mutex_parent };
    return m_parent;
}

HKTransformGraphNode* HK_API HKTransformGraphNodeImpl::internal_getChild(HKU32 idx)
{
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    if (m_children.size() > idx) { return m_children.at(idx); }
    else { return nullptr; }
}

const HKTransformGraphNode* HK_API HKTransformGraphNodeImpl::internal_getChild_const(HKU32 idx) const
{
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    if (m_children.size() > idx) { return m_children.at(idx); }
    else { return nullptr; }
}

void HK_API HKTransformGraphNodeImpl::destroyObject()
{
    ;
}

void HK_API HKTransformGraphNodeImpl::internal_popChild(Node* child)
{
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    auto iter = std::find(m_children.begin(), m_children.end(), child);
    if (iter != m_children.end()) {
        m_children.erase(iter);
        child->release();
    }
}

void HK_API HKTransformGraphNodeImpl::internal_addChild(Node* child)
{
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    auto iter = std::find(m_children.begin(), m_children.end(), child);
    if (iter == m_children.end()) {
        child->addRef();
        m_children.push_back(child);
    }

}

void HK_API HKTransformGraphNodeImpl::internal_update()
{
    auto transform_point_matrix  = m_parent_transform_point_matrix  * m_local_transform.transformPointMatrix();
    auto transform_vector_matrix = m_parent_transform_vector_matrix * m_local_transform.transformVectorMatrix();
    std::unique_lock<std::mutex> lk{ m_mutex_children };
    for (auto& child : m_children) {
        ((HKTransformGraphNodeImpl*)child)->internal_updateParent(transform_point_matrix, transform_vector_matrix);
    }
}

void HK_API HKTransformGraphNodeImpl::internal_updateParent(const HKMat4x4& new_parent_transform_point_matrix, const HKMat4x4& new_parent_transform_vector_matrix)
{
    std::unique_lock<std::mutex> lk_transform{ m_mutex_transform };
    auto rel_parent_transform_point_matrix  = new_parent_transform_point_matrix  * m_parent_transform_point_matrix.inverse();
    auto rel_parent_transform_vector_matrix = new_parent_transform_vector_matrix * m_parent_transform_vector_matrix.inverse();
    m_parent_transform_point_matrix  = new_parent_transform_point_matrix;
    m_parent_transform_vector_matrix = new_parent_transform_vector_matrix;
    auto children                           = internal_getChildrenRecursive();
    for (auto& child : children) {
        // �e�ɑΉ�����s����|���Z����
        ((HKTransformGraphNodeImpl*)child)->internal_mulMatrix(rel_parent_transform_point_matrix, rel_parent_transform_vector_matrix);
    }
}

HKTransformGraphImpl::HKTransformGraphImpl() :m_mutex{}, m_root_node{ HKTransformGraphNode_create() }
{
}

HK_API HKTransformGraphImpl::~HKTransformGraphImpl()
{
    if (m_root_node) {
        m_root_node->release();
    }
}

HKU32 HK_API HKTransformGraphImpl::addRef()
{
    return HKRefCntObject::addRef();
}

HKU32 HK_API HKTransformGraphImpl::release()
{
    return HKRefCntObject::release();
}

HKBool HK_API HKTransformGraphImpl::queryInterface(HKUUID iid, void** ppvInterface)
{
    if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_TransformGraph) {
        addRef();
        *ppvInterface = this;
        return true;
    }
    return false;
}

void HK_API HKTransformGraphImpl::setChild(HKTransformGraphNode* node)
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    if (!node ||!m_root_node ||m_root_node == node) { return; }
    node->setParent(m_root_node);
}

HKU32 HK_API HKTransformGraphImpl::getChildCount() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getChildCount(m_root_node);
}

void HK_API HKTransformGraphImpl::setObject(HKUnknown* object)
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setObject(m_root_node, object);
}

HKAffine3D HK_API HKTransformGraphImpl::getLocalTransform() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getLocalTransform(m_root_node);
}

void HK_API HKTransformGraphImpl::setLocalTransform(const HKAffine3D& transform)
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setLocalTransform(m_root_node, transform);
}
HKQuat HK_API HKTransformGraphImpl::getLocalRotation() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getLocalRotation(m_root_node);
}
HKVec3 HK_API HKTransformGraphImpl::getLocalPosition() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getLocalPosition(m_root_node);
}
HKVec3 HK_API HKTransformGraphImpl::getLocalScaling() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getLocalScaling(m_root_node);
}
void HK_API HKTransformGraphImpl::setLocalRotation(const HKQuat& rotation) {
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setLocalRotation(m_root_node, rotation);
}
void HK_API HKTransformGraphImpl::setLocalPosition(const HKVec3& position) {
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setLocalPosition(m_root_node, position);
}
void HK_API HKTransformGraphImpl::setLocalScaling(const HKVec3& scaling) {
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setLocalScaling(m_root_node, scaling);
}

HKVec3 HK_API HKTransformGraphImpl::getPosition() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getPosition(m_root_node);
}

void HK_API HKTransformGraphImpl::setPosition(const HKVec3& position)
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_setPosition(m_root_node, position);
}

HKMat4x4 HK_API HKTransformGraphImpl::getTransformPointMatrix() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getTransformPointMatrix(m_root_node);
}

HKMat4x4 HK_API HKTransformGraphImpl::getTransformVectorMatrix() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getTransformVectorMatrix(m_root_node);
}

HKVec3 HK_API HKTransformGraphImpl::transformPoint(const HKVec3& p) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_transformPoint(m_root_node, p);
}

HKVec3 HK_API HKTransformGraphImpl::transformVector(const HKVec3& v) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_transformVector(m_root_node, v);
}

HKVec3 HK_API HKTransformGraphImpl::transformDirection(const HKVec3& d) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_transformDirection(m_root_node, d);
}

HKVec3 HK_API HKTransformGraphImpl::inverseTransformPoint(const HKVec3& p) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_inverseTransformPoint(m_root_node, p);
}

HKVec3 HK_API HKTransformGraphImpl::inverseTransformVector(const HKVec3& v) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_inverseTransformVector(m_root_node, v);
}

HKVec3 HK_API HKTransformGraphImpl::inverseTransformDirection(const HKVec3& d) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_inverseTransformDirection(m_root_node, d);
}

HKUnknown* HK_API HKTransformGraphImpl::internal_getObject()
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getObject(m_root_node);
}

const HKUnknown* HK_API HKTransformGraphImpl::internal_getObject_const() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getObject_const(m_root_node);
}

const HKTransformGraphNode* HK_API HKTransformGraphImpl::internal_getRootNode_const() const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return m_root_node;
}

HKTransformGraphNode* HK_API HKTransformGraphImpl::internal_getChild(HKU32 idx)
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getChild(m_root_node,idx);
}

const HKTransformGraphNode* HK_API HKTransformGraphImpl::internal_getChild_const(HKU32 idx) const
{
    std::unique_lock<std::mutex> lk{ m_mutex };
    return HKTransformGraphNode_getChild_const(m_root_node, idx);
}

void HK_API HKTransformGraphImpl::destroyObject()
{
    return ;
}
