#pragma once
#include <hikari/core/node.h>
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // ShapeObject...形状を管理するためのObject
    // 
    struct ShapeObject : public Object {
      virtual ~ShapeObject() {}
      virtual auto getMinBBox() const->Vec3 = 0;
      virtual auto getMaxBBox() const->Vec3 = 0;
    };
    // FilterObject...シーンに対してShapeを追加するためのObject
    // 
    struct ShapeFilterObject   : public NodeComponentObject {
      virtual ~ShapeFilterObject() {}
    private:
      std::shared_ptr<ShapeObject> m_shape;
      std::weak_ptr<NodeObject> m_node;
    };
    // RenderObject...実際に描画対象となるObjectに対して設定する
    // 
    struct ShapeRenderObject   : public NodeComponentObject {
      virtual ~ShapeRenderObject() {}
      // Materialの設定が可能
    };
  }
}
