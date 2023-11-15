#include "node.h"

auto HKSceneNode::create() -> std::shared_ptr<HKSceneNode> { return std::shared_ptr<HKSceneNode>(new HKSceneNode()); }
HKSceneNode::~HKSceneNode() noexcept{}

HKSceneNode::HKSceneNode() noexcept{}