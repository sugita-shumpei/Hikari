#ifndef HK_SCENE_CAMERA__H
#define HK_SCENE_CAMERA__H

#include "node_component.h"
#include "../camera.h"
#include <memory>

struct HKSceneCamera : public HKSceneNodeComponent {
    std::shared_ptr<HKCamera> camera;
};

#endif
