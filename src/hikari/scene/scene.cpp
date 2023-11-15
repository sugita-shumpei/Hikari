#include "scene.h"

auto HKScene::create() -> std::shared_ptr<HKScene> { return std::shared_ptr<HKScene>(new HKScene()); }
HKScene::~HKScene() noexcept{}

auto HKScene::getRootNode() const -> std::shared_ptr<HKSceneNode> { return m_root_node; }

HKScene::HKScene() noexcept{}