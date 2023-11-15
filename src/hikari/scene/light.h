#ifndef HK_SCENE_LIGHT__H
#define HK_SCENE_LIGHT__H

#include <memory>

#include "node_component.h"
#include "../light.h"

struct HKSceneLight : public HKSceneNodeComponent {
    std::shared_ptr<HKLight> light;
};


#endif
