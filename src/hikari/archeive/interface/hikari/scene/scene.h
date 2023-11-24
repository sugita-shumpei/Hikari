#ifndef HK_SCENE_SCENE__H
#define HK_SCENE_SCENE__H

#include <memory>
#include "node.h"

struct HKScene {
    static auto create() -> std::shared_ptr<HKScene>;
    ~HKScene() noexcept;
    
    auto getRootNode() const -> std::shared_ptr<HKSceneNode> { return m_root_node; }
private:
    HKScene() noexcept;
private:
    std::shared_ptr<HKSceneNode> m_root_node = nullptr;
};

#endif
