#ifndef HK_SCENE_MATERIAL__H
#define HK_SCENE_MATERIAL__H

#include <memory>
#include "node_component.h"
#include "../bsdf.h"

struct HKSceneMaterial : public HKSceneNodeComponent {
    std::shared_ptr<HKBSDF> bsdf;
};


#endif
