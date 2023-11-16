#ifndef HK_TRANSFORM_GRAPH__H
#define HK_TRANSFORM_GRAPH__H
// Tranform Graph: �C�ӂ�HKObject�ɑ΂���g�����X�t�H�[�����܂ލ\���؂��\�z����
// Scene Node�̎����ɗ��p
#include "math/affine_3d.h"

#if defined(__cplusplus)
#include "object.h" 
#include "ref_cnt_object.h"
#endif
// {DC64DB21-B3C8-4261-8AF9-DCB913E20603}
#define HK_OBJECT_TYPEID_TransformGraphNode HK_UUID_DEFINE(0xdc64db21, 0xb3c8, 0x4261, 0x8a, 0xf9, 0xdc, 0xb9, 0x13, 0xe2, 0x6, 0x3)
#if defined(__cplusplus) 
struct HKTransformGraphNode : public HKUnknown{
    static HK_CXX11_CONSTEXPR HKUUID TypeID() HK_CXX_NOEXCEPT  { return HK_OBJECT_TYPEID_TransformGraphNode; }
    typedef HKTransformGraphNode Node;
    static HK_INLINE auto create() -> HKTransformGraphNode*;
    virtual void              HK_API setParent(HKTransformGraphNode* parent) = 0;
    virtual HKU32             HK_API getChildCount() const = 0;
    virtual void              HK_API setObject(HKUnknown* object) = 0;

    virtual HKUnknown*        HK_API internal_getObject() = 0;
    virtual const HKUnknown*  HK_API internal_getObject_const() const = 0;
    virtual Node*             HK_API internal_getParent() = 0;
    virtual const Node*       HK_API internal_getParent_const() const = 0;
    virtual Node*             HK_API internal_getChild(HKU32 idx) = 0;
    virtual const Node*       HK_API internal_getChild_const(HKU32 idx) const = 0;
    // transform
    virtual HKAffine3D        HK_API getLocalTransform()const = 0;
    virtual HKQuat            HK_API getLocalRotation() const = 0;
    virtual HKVec3            HK_API getLocalPosition() const = 0;
    virtual HKVec3            HK_API getLocalScaling()  const = 0;
    virtual void              HK_API setLocalTransform(const HKAffine3D& transform) = 0;
    virtual void              HK_API setLocalRotation(const HKQuat& rotation) = 0;
    virtual void              HK_API setLocalPosition(const HKVec3& position) = 0;
    virtual void              HK_API setLocalScaling(const HKVec3& scaling) = 0;
    virtual HKVec3            HK_API getPosition() const = 0;
    virtual void              HK_API setPosition(const HKVec3& position) = 0;
    virtual HKMat4x4          HK_API getTransformPointMatrix() const = 0;
    virtual HKMat4x4          HK_API getTransformVectorMatrix()const = 0;
    virtual HKVec3            HK_API transformPoint(const HKVec3& p) const = 0;
    virtual HKVec3            HK_API transformVector(const HKVec3& v) const = 0;
    virtual HKVec3            HK_API transformDirection(const HKVec3& d) const = 0;
    virtual HKVec3            HK_API inverseTransformPoint(const HKVec3& p) const = 0;
    virtual HKVec3            HK_API inverseTransformVector(const HKVec3& v) const = 0;
    virtual HKVec3            HK_API inverseTransformDirection(const HKVec3& d) const = 0;

    HK_INLINE HKUnknown*       getObject()       { return internal_getObject(); }
    HK_INLINE const HKUnknown* getObject() const { return internal_getObject_const(); }
    HK_INLINE Node*            getParent()       { return internal_getParent(); }
    HK_INLINE const Node*      getParent() const { return internal_getParent_const(); }
    HK_INLINE Node*            getChild(HKU32 idx)        { return internal_getChild(idx); }
    HK_INLINE const Node*      getChild(HKU32 idx)  const { return internal_getChild_const(idx); }
};
#else
typedef struct HKTransformGraphNode HKTransformGraphNode;
#endif
// {E1759A35-2880-4483-9F8D-77B78D2EFA01}
#define HK_OBJECT_TYPEID_TransformGraph HK_UUID_DEFINE(0xe1759a35, 0x2880, 0x4483, 0x9f, 0x8d, 0x77, 0xb7, 0x8d, 0x2e, 0xfa, 0x1)
#if defined(__cplusplus) 
struct HKTransformGraph : public HKUnknown  {
    static HK_CXX11_CONSTEXPR HKUUID TypeID()  HK_CXX_NOEXCEPT  { return HK_OBJECT_TYPEID_TransformGraph; }
    typedef HKTransformGraphNode Node;
    static HK_INLINE auto create() -> HKTransformGraph*;
    virtual HKU32             HK_API getChildCount() const = 0;
    virtual void              HK_API setObject(HKUnknown* object) = 0;
    virtual void              HK_API setChild(HKTransformGraphNode* rootNode) = 0;
    
    virtual HKUnknown*        HK_API internal_getObject() = 0;
    virtual const HKUnknown*  HK_API internal_getObject_const() const = 0;
    virtual const Node*       HK_API internal_getRootNode_const() const = 0;
    virtual Node*             HK_API internal_getChild(HKU32 idx) = 0;
    virtual const Node*       HK_API internal_getChild_const(HKU32 idx) const = 0;
    // transform
    virtual HKAffine3D        HK_API getLocalTransform() const = 0;
    virtual HKQuat            HK_API getLocalRotation() const = 0;
    virtual HKVec3            HK_API getLocalPosition() const = 0;
    virtual HKVec3            HK_API getLocalScaling()  const = 0;
    virtual void              HK_API setLocalTransform(const HKAffine3D& transform) = 0;
    virtual void              HK_API setLocalRotation(const HKQuat& rotation) = 0;
    virtual void              HK_API setLocalPosition(const HKVec3& position) = 0;
    virtual void              HK_API setLocalScaling(const HKVec3& scaling) = 0;
    virtual HKVec3            HK_API getPosition() const = 0;
    virtual void              HK_API setPosition(const HKVec3& position) = 0;
    virtual HKMat4x4          HK_API getTransformPointMatrix()const = 0;
    virtual HKMat4x4          HK_API getTransformVectorMatrix()const = 0;
    virtual HKVec3            HK_API transformPoint(const HKVec3& p) const = 0;
    virtual HKVec3            HK_API transformVector(const HKVec3& v) const = 0;
    virtual HKVec3            HK_API transformDirection(const HKVec3& d) const = 0;
    virtual HKVec3            HK_API inverseTransformPoint(const HKVec3& p) const = 0;
    virtual HKVec3            HK_API inverseTransformVector(const HKVec3& v) const = 0;
    virtual HKVec3            HK_API inverseTransformDirection(const HKVec3& d) const = 0;

    HK_INLINE HKUnknown*       getObject()       { return internal_getObject(); }
    HK_INLINE const HKUnknown* getObject() const { return internal_getObject_const(); }
    HK_INLINE const Node*      getRootNode() const { return internal_getRootNode_const(); }
    HK_INLINE Node*            getChild(HKU32 idx)        { return internal_getChild(idx); }
    HK_INLINE const Node*      getChild(HKU32 idx)  const { return internal_getChild_const(idx); }
};
#else
typedef struct HKTransformGraph HKTransformGraph;
#endif
// 
HK_EXTERN_C HK_DLL HKTransformGraphNode*       HK_API HKTransformGraphNode_create();
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setParent(HKTransformGraphNode* node, HKTransformGraphNode* parent);
HK_EXTERN_C HK_DLL HKU32                       HK_API HKTransformGraphNode_getChildCount(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setObject(HKTransformGraphNode* node, HKUnknown* object);
HK_EXTERN_C HK_DLL HKUnknown*                  HK_API HKTransformGraphNode_getObject(HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL const HKUnknown*            HK_API HKTransformGraphNode_getObject_const(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKTransformGraphNode*       HK_API HKTransformGraphNode_getParent(HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL const HKTransformGraphNode* HK_API HKTransformGraphNode_getParent_const(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKTransformGraphNode*       HK_API HKTransformGraphNode_getChild(HKTransformGraphNode* node,HKU32 idx);
HK_EXTERN_C HK_DLL const HKTransformGraphNode* HK_API HKTransformGraphNode_getChild_const(const HKTransformGraphNode* node,HKU32 idx);
HK_EXTERN_C HK_DLL HKCAffine3D                 HK_API HKTransformGraphNode_getLocalTransform(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKCQuat                     HK_API HKTransformGraphNode_getLocalRotation(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_getLocalPosition(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_getLocalScaling(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setLocalTransform(HKTransformGraphNode* node, HKCAffine3D transform);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setLocalRotation(HKTransformGraphNode* node, HKCQuat     rotation);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setLocalPosition(HKTransformGraphNode* node, HKCVec3     position);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setLocalScaling(HKTransformGraphNode* node, HKCVec3     scaling);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_getPosition(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraphNode_setPosition(HKTransformGraphNode* node, HKCVec3     position);
HK_EXTERN_C HK_DLL HKCMat4x4                   HK_API HKTransformGraphNode_getTransformPointMatrix(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKCMat4x4                   HK_API HKTransformGraphNode_getTransformVectorMatrix(const HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_transformPoint(HKTransformGraphNode* node, HKCVec3 p);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_transformVector(HKTransformGraphNode* node, HKCVec3 v);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_transformDirection(HKTransformGraphNode* node, HKCVec3 d);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_inverseTransformPoint(HKTransformGraphNode* node, HKCVec3 p);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_inverseTransformVector(HKTransformGraphNode* node, HKCVec3 v);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraphNode_inverseTransformDirection(HKTransformGraphNode* node, HKCVec3 d);

HK_EXTERN_C HK_DLL HKTransformGraph*           HK_API HKTransformGraph_create();
HK_EXTERN_C HK_DLL HKU32                       HK_API HKTransformGraph_getChildCount(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setObject(HKTransformGraph* root, HKUnknown* object);
HK_EXTERN_C HK_DLL HKUnknown*                  HK_API HKTransformGraph_getObject(HKTransformGraph* root);
HK_EXTERN_C HK_DLL const HKUnknown*            HK_API HKTransformGraph_getObject_const(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL const HKTransformGraphNode* HK_API HKTransformGraph_getRootNode_const(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setChild(HKTransformGraph* root, HKTransformGraphNode* node);
HK_EXTERN_C HK_DLL HKTransformGraphNode*       HK_API HKTransformGraph_getChild(HKTransformGraph* root,HKU32 idx);
HK_EXTERN_C HK_DLL const HKTransformGraphNode* HK_API HKTransformGraph_getChild_const(const HKTransformGraph* root,HKU32 idx);
HK_EXTERN_C HK_DLL HKCAffine3D                 HK_API HKTransformGraph_getLocalTransform(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL HKCQuat                     HK_API HKTransformGraph_getLocalRotation(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraph_getLocalPosition(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraph_getLocalScaling(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setLocalTransform(HKTransformGraph* root, HKCAffine3D transform);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setLocalRotation(HKTransformGraph*  root, HKCQuat     rotation);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setLocalPosition(HKTransformGraph*  root, HKCVec3     position);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setLocalScaling(HKTransformGraph*   root, HKCVec3     scaling);
HK_EXTERN_C HK_DLL HKCVec3                     HK_API HKTransformGraph_getPosition(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL void                        HK_API HKTransformGraph_setPosition(HKTransformGraph* root, HKCVec3     position);
HK_EXTERN_C HK_DLL HKCMat4x4                  HK_API HKTransformGraph_getTransformPointMatrix(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL HKCMat4x4                  HK_API HKTransformGraph_getTransformVectorMatrix(const HKTransformGraph* root);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_transformPoint(HKTransformGraph* root, HKCVec3 p);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_transformVector(HKTransformGraph* root, HKCVec3 v);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_transformDirection(HKTransformGraph* root, HKCVec3 d);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_inverseTransformPoint(HKTransformGraph* root, HKCVec3 p);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_inverseTransformVector(HKTransformGraph* root, HKCVec3 v);
HK_EXTERN_C HK_DLL HKCVec3                    HK_API HKTransformGraph_inverseTransformDirection(HKTransformGraph* root, HKCVec3 d);

#if defined(__cplusplus) 
HK_INLINE auto HKTransformGraphNode::create() -> HKTransformGraphNode* { return HKTransformGraphNode_create(); }
HK_INLINE auto HKTransformGraph::create() -> HKTransformGraph* { return HKTransformGraph_create(); }
#endif

#endif
