#ifndef HK_SCENE_NODE__H
#define HK_SCENE_NODE__H

#include <memory>
#include "node_component.h"

struct HKSceneNode {
    static auto create() -> std::shared_ptr<HKSceneNode>;
    virtual ~HKSceneNode() noexcept;
private:
    HKSceneNode()noexcept;
private:

};

#endif
