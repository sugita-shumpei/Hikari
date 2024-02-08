#include <hikari/core/node.h>

auto hikari::core::NodeObject::create(const Str& name, const Transform& transform) -> std::shared_ptr<NodeObject> {
  auto res         = std::shared_ptr<NodeObject>(new NodeObject(name, transform));
  auto tra         = std::shared_ptr<NodeTransformObject>(new NodeTransformObject(res));
  res->m_transform = tra;
  return res;
}

auto hikari::core::NodeObject::getPropertyNames() const -> std::vector<Str>
{
  std::vector<Str> res = {};
  res.push_back("name");
  res.push_back("parent");
  res.push_back("children");
  res.push_back("child_count");
  res.push_back("global_transform");
  res.push_back("global_matrix");
  res.push_back("global_position");
  res.push_back("global_rotation");
  res.push_back("global_scale");
  res.push_back("local_transform");
  res.push_back("local_matrix");
  res.push_back("local_position");
  res.push_back("local_rotation");
  res.push_back("local_scale");
  return res;
}

void hikari::core::NodeObject::setPropertyBlock(const PropertyBlock& pb)
{
  auto name = pb.getValue("name");
  if (name.getTypeIndex() == PropertyTypeIndex<Str>::value) {
    setName(*name.getValue<Str>());
  }
  auto parent = pb.getValue("parent");
  if (parent.getTypeIndex() == PropertyTypeIndex<std::shared_ptr<Object>>::value) {
    auto parent_object = parent.getValue<Node>();
    if (parent_object) {
      setParent(parent_object.getObject());
    }
    else {
      setParent({});
    }
  }
  auto children = pb.getValue("children");
  if (children.getTypeIndex() == PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value) {
    popChildren();
    auto child_objects = children.getValue<Array<Node>>();
    for (auto& child_object : child_objects) {
      child_object.getObject()->setParent(shared_from_this());
    }
  }

  auto transform = pb.getValue("transform");
  if (transform.getTypeIndex() == PropertyTypeIndex<Transform>::value) {
    setLocalTransform(*transform.getValue<Transform>());
  }
}

void hikari::core::NodeObject::getPropertyBlock(PropertyBlock& pb) const
{
  pb.clear();
  auto children = std::vector<Node>();
  children.reserve(m_children.size());
  std::transform(m_children.begin(), m_children.end(), std::back_inserter(children), [](const auto& p) { return Node(p); });
  pb.setValue("name"     , getName());
  pb.setValue("parent"   , Node(getParent()));
  pb.setValue("children" , children);
  pb.setValue("transform", m_local_transform);
}

hikari::Bool hikari::core::NodeObject::hasProperty(const Str& name) const
{
  if (name == "name") { return true; }
  if (name == "parent") { return true; }
  if (name == "children") { return true; }
  if (name == "child_count") { return true; }
  if (name == "global_transform") { return true; }
  if (name == "global_matrix") { return true; }
  if (name == "global_position") { return true; }
  if (name == "global_rotation") { return true; }
  if (name == "global_scale") { return true; }
  if (name == "local_transform") { return true; }
  if (name == "local_matirx") { return true; }
  if (name == "local_position") { return true; }
  if (name == "local_rotation") { return true; }
  if (name == "local_scale") { return true; }
  return false;
}

hikari::Bool hikari::core::NodeObject::setProperty(const Str& name, const Property& value)
{
  if (name == "name") {
    auto str = value.getValue<Str>();
    if (str) {
      setName(*str);
      return true;
    }
    return false;
  }
  if (name == "parent") {
    if (value.getTypeIndex() == PropertyTypeIndex<std::shared_ptr<Object>>::value) {
      auto parent = value.getValue<Node>();
      if (!parent) {
        setParent(nullptr);
      }
      else {
        setParent(parent.getObject());
      }
      return true;
    }
    else if (value.getTypeIndex() == PropertyTypeIndex<void>::value) {
      setParent(nullptr);
      return true;
    }
    return false;
  }
  if (name == "child_count") {
    auto childCount = value.toU64();
    if (childCount) { setChildCount(*childCount); }
    return true;
  }
  if (name == "global_transform") {
    auto global_transform = value.getValue<Transform>();
    if (global_transform) {
      setGlobalTransform(*global_transform);
      return true;
    }
    return false;
  }
  if (name == "local_transform") {
    auto local_transform = value.getValue<Transform>();
    if (local_transform) {
      setLocalTransform(*local_transform);
      return true;
    }
    return false;
  }
  if (name == "global_matrix") {
    auto global_matrix = value.toMat();
    if (global_matrix) { setGlobalTransform(Transform(*global_matrix)); return true; }
    return false;
  }
  if (name == "local_matirx") {
    auto local_matrix = value.toMat();
    if (local_matrix) { setLocalTransform(Transform(*local_matrix)); return true; }
    return false;
  }
  return false;
}

hikari::Bool hikari::core::NodeObject::getProperty(const Str& name, Property& value) const
{
  if (name == "name") {
    value = getName();
    return true;
  }
  if (name == "parent") {
    value = Node(m_parent.lock());
    return true;
  }
  if (name == "children") {
    auto children = getChildren();
    auto res = Array<Node>();
    res.reserve(children.size());
    std::transform(std::begin(children), std::end(children), std::back_inserter(res), [](const auto& p) {
      return Node(p);
    });
    value = res;
    return true;
  }
  if (name == "child_count") {
    value = (U64)getChildCount();
    return true;
  }
  if (name == "global_transform") {
    value = getGlobalTransform();
    return true;
  }
  if (name == "local_transform") {
    value = getLocalTransform();
    return true;
  }
  if (name == "global_matrix") {
    value = getGlobalMatrix();
    return true;
  }
  if (name == "local_matirx") {
    value = getLocalMatrix();
    return true;
  }
  if (name == "global_position") {
    value = getGlobalPosition();
    return true;
  }
  if (name == "local_position") {
    value = getLocalPosition();
    return true;
  }
  if (name == "global_rotation") {
    value = getGlobalRotation();
    return true;
  }
  if (name == "local_rotation") {
    value = getLocalRotation();
    return true;
  }
  if (name == "global_scale") {
    value = getGlobalScale();
    return true;
  }
  if (name == "local_scale") {
    value = getLocalScale();
    return true;
  }
  return false;
}

hikari::Bool hikari::core::NodeObject::hasComponent(const Str& componentType) const {
  return std::find_if(std::begin(m_components), std::end(m_components), [&componentType](const auto& v) { return v->isConvertible(componentType); }) != std::end(m_components);
}

void hikari::core::NodeObject::popComponent(const Str& componentType) {
  auto iter = std::find_if(std::begin(m_components), std::end(m_components), [&componentType](const auto& v) { return v->isConvertible(componentType); });
  if (iter != std::end(m_components)) {
    auto res = *iter;
    m_components.erase(iter);
  }
}

auto hikari::core::NodeObject::getComponent(const Str& componentType) const -> std::shared_ptr<NodeComponentObject> {
  if (componentType == NodeObject::TypeString()) { return m_transform; }
  if (componentType == NodeTransformObject::TypeString()) { return m_transform; }
  auto iter = std::find_if(std::begin(m_components), std::end(m_components), [&componentType](const auto& v) {  return v->isConvertible(componentType); });
  if (iter != std::end(m_components)) {
    auto res = *iter;
    return res;
  }
  return nullptr;
}

auto hikari::core::NodeObject::getComponentInParent(const Str& componentType) const -> std::shared_ptr<NodeComponentObject> {
  auto parent = getParent();
  if (!parent) { return nullptr; }
  auto res = parent->getComponent(componentType);
  if (res) { return res; }
  return parent->getComponentInParent(componentType);
}

auto hikari::core::NodeObject::getComponentInChildren(const Str& componentType) const -> std::shared_ptr<NodeComponentObject> {
  for (auto& child : m_children) {
    auto res = child->getComponent(componentType);
    if (res) { return res; }
  }
  for (auto& child : m_children) {
    auto res = child->getComponentInChildren(componentType);
    if (res) { return res; }
  }
  return nullptr;
}

auto hikari::core::NodeObject::getComponents(const Str& componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>> {
  if (componentType == NodeTransformObject::TypeString()) {
    return { m_transform };
  }
  auto res = std::vector<std::shared_ptr<NodeComponentObject>>();
  if (componentType == NodeObject::TypeString()) { res.push_back(m_transform); }
  for (auto& component : m_components) {
    if (component->isConvertible(componentType)) {
      res.push_back(component);
    }
  }
  return res;
}

auto hikari::core::NodeObject::getComponentsInParent(const Str& componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>> {
  auto parent = getParent();
  if (!parent) { return {}; }
  auto res = parent->getComponentsInChildren(componentType);
  auto tmp = parent->getComponentsInParent(componentType);
  res.reserve(res.size() + tmp.size());
  std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(res));
  return res;
}

auto hikari::core::NodeObject::getComponentsInChildren(const Str& componentType) const -> std::vector<std::shared_ptr<NodeComponentObject>> {
  auto res = std::vector<std::shared_ptr<NodeComponentObject>>();
  for (auto& child : m_children) {
    auto tmp = child->getComponents(componentType);
    res.reserve(res.size() + tmp.size());
    std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(res));
  }
  for (auto& child : m_children) {
    auto tmp = child->getComponentsInChildren(componentType);
    res.reserve(res.size() + tmp.size());
    std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(res));
  }
  return res;
}

auto hikari::core::NodeObject::getName() const -> Str { return m_name; }

void hikari::core::NodeObject::setName(const Str& name) { m_name = name; }

auto hikari::core::NodeObject::getParent() const -> std::shared_ptr<NodeObject> { return m_parent.lock(); }

void hikari::core::NodeObject::setParent(const std::shared_ptr<NodeObject>& new_parent) {
  auto old_parent = m_parent.lock();
  if (!old_parent) {
    if (!new_parent) { return; }
    m_parent = new_parent;
    new_parent->m_children.push_back(shared_from_this());
    auto new_parent_transform = new_parent->getGlobalTransform();
    updateParentGlobalTransform(new_parent_transform);
  }
  else {
    auto iter = std::find(std::begin(old_parent->m_children), std::end(old_parent->m_children), shared_from_this());
    if (iter != std::end(old_parent->m_children)) {
      old_parent->m_children.erase(iter);
    }
    if (!new_parent) {
      m_parent = {};
      updateParentGlobalTransform(Transform());
      return;
    }
    else {
      m_parent = new_parent;
      new_parent->m_children.push_back(shared_from_this());
      auto new_parent_transform = new_parent->getGlobalTransform();
      updateParentGlobalTransform(new_parent_transform);
    }
  }
}

auto hikari::core::NodeObject::getChildren() const -> const std::vector<std::shared_ptr<NodeObject>>& { return m_children; }

void hikari::core::NodeObject::setChildren(const std::vector<std::shared_ptr<NodeObject>>& children)
{
}

void hikari::core::NodeObject::popChildren()
{
  for (auto& child : m_children) {
    child->m_parent = {};
    child->updateParentGlobalTransform(Transform());
  }
  m_children.clear();
}

auto hikari::core::NodeObject::getChild(size_t idx) const -> std::shared_ptr<NodeObject>
{
  if (m_children.size() > idx) { return m_children[idx]; }
  return std::shared_ptr<NodeObject>();
}

void hikari::core::NodeObject::setChild(size_t idx, const std::shared_ptr<NodeObject>& child)
{
  if (m_children.size() > idx) {
    if (!child) {
      popChild(idx);
      return;
    }
    auto new_parent = shared_from_this();
    auto old_parent = child->m_parent.lock();
    if (new_parent == old_parent) { return; }
    if (old_parent) {
      auto iter = std::find(std::begin(old_parent->m_children), std::end(old_parent->m_children), child);
      old_parent->m_children.erase(iter);
    }
    m_children[idx] = child;
    child->m_parent = new_parent;
    child->updateParentGlobalTransform(getGlobalTransform());
    return;
  }
}

void hikari::core::NodeObject::addChild(const std::shared_ptr<NodeObject>& child)
{
  auto parent = shared_from_this();
  if (!parent || !child) { return; }
  child->setParent(parent);
}

void hikari::core::NodeObject::popChild(size_t idx)
{
  if (m_children.size() > idx) {
    auto iter_child = m_children.begin() + idx;
    m_children.erase(iter_child);
    auto child = (*iter_child);
    child->m_parent = {};
    child->updateParentGlobalTransform(Transform());
  }
}

auto hikari::core::NodeObject::getChildCount() const -> size_t
{
  return m_children.size();
}

void hikari::core::NodeObject::setChildCount(size_t count)
{
  size_t old_size = m_children.size();
  if (count <= old_size) {
    for (size_t i = count; i < old_size; ++i) {
      auto child = m_children[i];
      child->m_parent = {};
      child->updateParentGlobalTransform(Transform());
    }
    m_children.resize(count);
  }
  else {
    for (size_t i = old_size; i < count; ++i) {
      auto child = NodeObject::create("", {});
      child->setParent(shared_from_this());
    }
  }
}

void hikari::core::NodeObject::setGlobalTransform(const Transform& transform) {
  auto rel_transform = m_global_transform.inverse() * transform;
  m_global_transform = transform;// global transform
  auto parent = getParent();
  if (parent) {
    m_local_transform = m_global_transform * parent->getGlobalTransform().inverse();
  }
  else {
    m_local_transform = m_global_transform;
  }
  updateRelGlobalTransform(rel_transform);
}

void hikari::core::NodeObject::getGlobalTransform(Transform& transform) const {
  transform = m_global_transform;
}

auto hikari::core::NodeObject::getGlobalTransform() const -> Transform {
  return m_global_transform;
}

auto hikari::core::NodeObject::getGlobalMatrix() const -> glm::mat4 { return m_global_transform.getMat(); }

bool hikari::core::NodeObject::getGlobalPosition(Vec3& position) {
  auto position_ = m_global_transform.getPosition();
  if (position_) { position = *position_; return true; }
  return false;
}

bool hikari::core::NodeObject::getGlobalRotation(Quat& rotation) {
  auto rotation_ = m_global_transform.getRotation();
  if (rotation_) { rotation = *rotation_; return true; }
  return false;
}

bool hikari::core::NodeObject::getGlobalScale(Vec3& scale) {
  auto scale_ = m_global_transform.getScale();
  if (scale_) { scale = *scale_; return true; }
  return false;
}

auto hikari::core::NodeObject::getGlobalPosition() const -> Option<Vec3> { return m_global_transform.getPosition(); }

auto hikari::core::NodeObject::getGlobalRotation() const -> Option<Quat> { return m_global_transform.getRotation(); }

auto hikari::core::NodeObject::getGlobalScale() const -> Option<Vec3> { return m_global_transform.getScale(); }

void hikari::core::NodeObject::setLocalTransform(const Transform& transform) {
  m_local_transform = transform;
  auto parent = getParent();
  auto old_global_transform = m_global_transform;
  if (parent) {
    m_global_transform = m_local_transform * parent->getGlobalTransform();
  }
  else {
    m_global_transform = m_local_transform;
  }
  updateRelGlobalTransform(old_global_transform.inverse() * m_global_transform);
}

void hikari::core::NodeObject::getLocalTransform(Transform& transform) const {
  transform = m_local_transform;
}

auto hikari::core::NodeObject::getLocalTransform() const -> Transform { return m_local_transform; }

auto hikari::core::NodeObject::getLocalMatrix() const -> glm::mat4 { return m_local_transform.getMat(); }

bool hikari::core::NodeObject::getLocalPosition(Vec3& position) {
  auto position_ = m_local_transform.getPosition();
  if (position_) { position = *position_; return true; }
  return false;
}

bool hikari::core::NodeObject::getLocalRotation(Quat& rotation) {
  auto rotation_ = m_local_transform.getRotation();
  if (rotation_) { rotation = *rotation_; return true; }
  return false;
}

bool hikari::core::NodeObject::getLocalScale(Vec3& scale) {
  auto scale_ = m_local_transform.getScale();
  if (scale_) { scale = *scale_; return true; }
  return false;
}

auto hikari::core::NodeObject::getLocalPosition() const -> Option<Vec3> { return m_local_transform.getPosition(); }

auto hikari::core::NodeObject::getLocalRotation() const -> Option<Quat> { return m_local_transform.getRotation(); }

auto hikari::core::NodeObject::getLocalScale() const -> Option<Vec3> { return m_local_transform.getScale(); }


auto hikari::core::Node::operator[](size_t idx) const -> Node
{
  return Node(getChild(idx));
}

auto hikari::core::Node::operator[](size_t idx) -> NodeRef
{
  auto object = getObject();
  return NodeRef(object, idx);
}

auto hikari::core::Node::getSize() const -> size_t
{
  return getChildCount();
}

void hikari::core::Node::setSize(size_t count)
{
  setChildCount(count);
}

void hikari::core::Node::setName(const Str& name)
{
  auto object = getObject();
  if (object) {
    return object->setName(name);
  }
}

auto hikari::core::Node::getChildCount() const -> size_t
{
  auto object = getObject();
  if (object) {
    return object->getChildCount();
  }
  else {
    return 0;
  }
}

void hikari::core::Node::setChildCount(size_t count)
{
  auto object = getObject();
  if (object) {
    return object->setChildCount(count);
  }
}

auto hikari::core::Node::getChildren() const -> std::vector<Node>
{
  auto object = getObject();
  if (object) {
    auto children = object->getChildren();
    auto nodes = std::vector<Node>();
    nodes.reserve(children.size());
    std::transform(std::begin(children), std::end(children), std::back_inserter(nodes), [](const auto& obj) {
      return Node(obj);
      }
    );
    return nodes;
  }
  else {
    return {};
  }
}

void hikari::core::Node::setChildren(const std::vector<Node>& children)
{
  auto object = getObject();
  if (object) {
    auto tmp = std::vector<std::shared_ptr<NodeObject>>();
    for (auto& child : children) {
      tmp.push_back(child.getObject());
    }
    object->setChildren(tmp);
  }
}

void hikari::core::Node::popChildren()
{
  auto object = getObject();
  if (object) {
    object->popChildren();
  }

}

auto hikari::core::Node::getChild(size_t idx) const -> Node
{
  auto object = getObject();
  if (!object) { return Node(); }
  return Node(object->getChild(idx));
}

void hikari::core::Node::setChild(size_t idx, const Node& child)
{
  auto object = getObject();
  if (!object) { return; }
  object->setChild(idx, child.getObject());
}

void hikari::core::Node::addChild(const Node& child)
{
  auto object = getObject();
  if (!object) { return; }
  object->addChild(child.getObject());
}

void hikari::core::Node::popChild(size_t idx)
{
  auto object = getObject();
  if (!object) { return; }
  object->popChild(idx);
}

void hikari::core::Node::setGlobalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setGlobalTransform(transform);
}

void hikari::core::Node::getGlobalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getGlobalTransform(transform);
}

auto hikari::core::Node::getGlobalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getGlobalTransform();
}

auto hikari::core::Node::getGlobalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getGlobalMatrix();
}

bool hikari::core::Node::getGlobalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalPosition(position);
}

bool hikari::core::Node::getGlobalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalRotation(rotation);
}

bool hikari::core::Node::getGlobalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalScale(scale);
}

auto hikari::core::Node::getGlobalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalPosition();
}

auto hikari::core::Node::getGlobalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalRotation();
}

auto hikari::core::Node::getGlobalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalScale();
}

void hikari::core::Node::setLocalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setLocalTransform(transform);
}

void hikari::core::Node::getLocalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getLocalTransform(transform);
}

auto hikari::core::Node::getLocalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getLocalTransform();
}

auto hikari::core::Node::getLocalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getLocalMatrix();
}

bool hikari::core::Node::getLocalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalPosition(position);
}

bool hikari::core::Node::getLocalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalRotation(rotation);
}

bool hikari::core::Node::getLocalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalScale(scale);
}

auto hikari::core::Node::getLocalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalPosition();
}

auto hikari::core::Node::getLocalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalRotation();
}

auto hikari::core::Node::getLocalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalScale();
}

auto hikari::core::NodeRef::getChildCount() const -> size_t
{
  auto object = getObject();
  if (!object) { return 0; }
  return object->getChildCount();
}

void hikari::core::NodeRef::setChildCount(size_t count)
{
  auto object = getObject();
  if (!object) { return; }
  return object->setChildCount(count);
}

auto hikari::core::NodeRef::getChildren() const -> std::vector<Node>
{
  auto object = getObject();
  if (!object) { return {}; }
  auto children = object->getChildren();
  auto nodes = std::vector<Node>();
  nodes.reserve(children.size());
  std::transform(std::begin(children), std::end(children), std::back_inserter(nodes), [](const auto& obj) {
    return Node(obj);
    }
  );
  return nodes;
}

void hikari::core::NodeRef::setChildren(const std::vector<Node>& nodes)
{
  auto object = getObject();
  if (!object) { return; }
  auto children = std::vector<std::shared_ptr<NodeObject>>();
  children.reserve(nodes.size());
  std::transform(std::begin(nodes), std::end(nodes), std::back_inserter(children), [](const auto& obj) {
    return obj.getObject();
    }
  );
  object->setChildren(children);
}

auto hikari::core::NodeRef::operator[](size_t idx) const -> Node
{
  return getChild(idx);
}

auto hikari::core::NodeRef::operator[](size_t idx) -> Ref
{
  return NodeRef(getObject(), idx);
}

auto hikari::core::NodeRef::getSize() const -> size_t
{
  return getChildCount();
}

void hikari::core::NodeRef::setSize(size_t count)
{
  setChildCount(count);
}

void hikari::core::NodeRef::setName(const Str& name)
{
  auto object = getObject();
  if (object) {
    return object->setName(name);
  }
}

void hikari::core::NodeRef::popChildren()
{
  auto object = getObject();
  if (object) {
    object->popChildren();
  }
}

auto hikari::core::NodeRef::getChild(size_t idx) const -> Node
{
  auto object = getObject();
  return NodeRef(object, idx);
}

void hikari::core::NodeRef::setChild(size_t idx, const Node& child)
{
  auto object = getObject();
  if (!object) { return; }
  object->setChild(idx, child.getObject());
}

void hikari::core::NodeRef::addChild(const Node& child)
{
  auto object = getObject();
  if (!object) { return; }
  object->addChild(child.getObject());
}

void hikari::core::NodeRef::popChild(size_t idx)
{
  auto object = getObject();
  if (!object) { return; }
  object->popChild(idx);
}


void hikari::core::NodeRef::setGlobalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setGlobalTransform(transform);
}

void hikari::core::NodeRef::getGlobalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getGlobalTransform(transform);
}

auto hikari::core::NodeRef::getGlobalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getGlobalTransform();
}

auto hikari::core::NodeRef::getGlobalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getGlobalMatrix();
}

bool hikari::core::NodeRef::getGlobalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalPosition(position);
}

bool hikari::core::NodeRef::getGlobalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalRotation(rotation);
}

bool hikari::core::NodeRef::getGlobalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalScale(scale);
}

auto hikari::core::NodeRef::getGlobalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalPosition();
}

auto hikari::core::NodeRef::getGlobalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalRotation();
}

auto hikari::core::NodeRef::getGlobalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalScale();
}

void hikari::core::NodeRef::setLocalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setLocalTransform(transform);
}

void hikari::core::NodeRef::getLocalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getLocalTransform(transform);
}

auto hikari::core::NodeRef::getLocalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getLocalTransform();
}

auto hikari::core::NodeRef::getLocalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getLocalMatrix();
}

bool hikari::core::NodeRef::getLocalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalPosition(position);
}

bool hikari::core::NodeRef::getLocalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalRotation(rotation);
}

bool hikari::core::NodeRef::getLocalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalScale(scale);
}

auto hikari::core::NodeRef::getLocalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalPosition();
}

auto hikari::core::NodeRef::getLocalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalRotation();
}

auto hikari::core::NodeRef::getLocalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalScale();
}

hikari::core::NodeComponentObject::~NodeComponentObject() noexcept {}

auto hikari::core::NodeComponentObject::getName() const -> Str { auto node = getNode(); if (node) { return node->getName(); }  return ""; }



void hikari::core::NodeTransformObject::setGlobalTransform(const Transform& transform)
{
  auto object = getNode();
  if (!object) { return; }
  object->setGlobalTransform(transform);
}

void hikari::core::NodeTransformObject::getGlobalTransform(Transform& transform) const
{
  auto object = getNode();
  if (!object) { return; }
  object->getGlobalTransform(transform);
}

auto hikari::core::NodeTransformObject::getGlobalTransform() const -> Transform
{
  auto object = getNode();
  if (!object) { return Transform(); }
  return object->getGlobalTransform();
}

auto hikari::core::NodeTransformObject::getGlobalMatrix() const -> Mat4
{
  auto object = getNode();
  if (!object) { return Mat4(); }
  return object->getGlobalMatrix();
}

bool hikari::core::NodeTransformObject::getGlobalPosition(Vec3& position)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getGlobalPosition(position);
}

bool hikari::core::NodeTransformObject::getGlobalRotation(Quat& rotation)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getGlobalRotation(rotation);
}

bool hikari::core::NodeTransformObject::getGlobalScale(Vec3& scale)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getGlobalScale(scale);
}

auto hikari::core::NodeTransformObject::getGlobalPosition() const -> Option<Vec3>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getGlobalPosition();
}

auto hikari::core::NodeTransformObject::getGlobalRotation() const -> Option<Quat>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getGlobalRotation();
}

auto hikari::core::NodeTransformObject::getGlobalScale() const -> Option<Vec3>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getGlobalScale();
}

void hikari::core::NodeTransformObject::setLocalTransform(const Transform& transform)
{
  auto object = getNode();
  if (!object) { return; }
  object->setLocalTransform(transform);
}

void hikari::core::NodeTransformObject::getLocalTransform(Transform& transform) const
{
  auto object = getNode();
  if (!object) { return; }
  object->getLocalTransform(transform);
}

auto hikari::core::NodeTransformObject::getLocalTransform() const -> Transform
{
  auto object = getNode();
  if (!object) { return Transform(); }
  return object->getLocalTransform();
}

auto hikari::core::NodeTransformObject::getLocalMatrix() const -> Mat4
{
  auto object = getNode();
  if (!object) { return Mat4(); }
  return object->getLocalMatrix();
}

bool hikari::core::NodeTransformObject::getLocalPosition(Vec3& position)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getLocalPosition(position);
}

bool hikari::core::NodeTransformObject::getLocalRotation(Quat& rotation)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getLocalRotation(rotation);
}

bool hikari::core::NodeTransformObject::getLocalScale(Vec3& scale)
{
  auto object = getNode();
  if (!object) { return false; }
  return object->getLocalScale(scale);
}

auto hikari::core::NodeTransformObject::getLocalPosition() const -> Option<Vec3>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getLocalPosition();
}

auto hikari::core::NodeTransformObject::getLocalRotation() const -> Option<Quat>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getLocalRotation();
}

auto hikari::core::NodeTransformObject::getLocalScale() const -> Option<Vec3>
{
  auto object = getNode();
  if (!object) { return std::nullopt; }
  return object->getLocalScale();
}


void hikari::core::NodeTransform::setGlobalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setGlobalTransform(transform);
}

void hikari::core::NodeTransform::getGlobalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getGlobalTransform(transform);
}

auto hikari::core::NodeTransform::getGlobalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getGlobalTransform();
}

auto hikari::core::NodeTransform::getGlobalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getGlobalMatrix();
}

bool hikari::core::NodeTransform::getGlobalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalPosition(position);
}

bool hikari::core::NodeTransform::getGlobalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalRotation(rotation);
}

bool hikari::core::NodeTransform::getGlobalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getGlobalScale(scale);
}

auto hikari::core::NodeTransform::getGlobalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalPosition();
}

auto hikari::core::NodeTransform::getGlobalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalRotation();
}

auto hikari::core::NodeTransform::getGlobalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getGlobalScale();
}

void hikari::core::NodeTransform::setLocalTransform(const Transform& transform)
{
  auto object = getObject();
  if (!object) { return; }
  object->setLocalTransform(transform);
}

void hikari::core::NodeTransform::getLocalTransform(Transform& transform) const
{
  auto object = getObject();
  if (!object) { return; }
  object->getLocalTransform(transform);
}

auto hikari::core::NodeTransform::getLocalTransform() const -> Transform
{
  auto object = getObject();
  if (!object) { return Transform(); }
  return object->getLocalTransform();
}

auto hikari::core::NodeTransform::getLocalMatrix() const -> Mat4
{
  auto object = getObject();
  if (!object) { return Mat4(); }
  return object->getLocalMatrix();
}

bool hikari::core::NodeTransform::getLocalPosition(Vec3& position)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalPosition(position);
}

bool hikari::core::NodeTransform::getLocalRotation(Quat& rotation)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalRotation(rotation);
}

bool hikari::core::NodeTransform::getLocalScale(Vec3& scale)
{
  auto object = getObject();
  if (!object) { return false; }
  return object->getLocalScale(scale);
}

auto hikari::core::NodeTransform::getLocalPosition() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalPosition();
}

auto hikari::core::NodeTransform::getLocalRotation() const -> Option<Quat>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalRotation();
}

auto hikari::core::NodeTransform::getLocalScale() const -> Option<Vec3>
{
  auto object = getObject();
  if (!object) { return std::nullopt; }
  return object->getLocalScale();
}

hikari::core::NodeSerializer::~NodeSerializer() noexcept
{
}

 auto hikari::core::NodeSerializer::getTypeString() const noexcept -> Str
{
  return NodeObject::TypeString();
}

 auto hikari::core::NodeSerializer::eval(const std::shared_ptr<Object>& object)const -> Json
{
  if (!object) { return Json(); }
  auto node          = Node(std::static_pointer_cast<NodeObject>(object));
  Json json          = {};
  json["type"]       = "Node";
  json["name"]       = object->getName();
  json["properties"] = {};
  auto children      = node.getChildren();
  json["properties"]["children"] = Array<Json>();
  for (auto& child : children) {
    json["properties"]["children"].push_back(eval(child.getObject()));
  }
  json["properties"]["transform"]  = PropertySerializer::eval(Property(node.getLocalTransform()));
  json["properties"]["components"] = Array<Json>();
  auto components = node.getComponents<NodeComponent>();
  {
    size_t i = 0;
    for (auto& component : components) {
      if (i != 0) {
        auto tmp = ObjectSerializeManager::getInstance().serialize(component.getObject());
        if (tmp) {
          json["properties"].push_back(tmp);
        }
      }
      ++i;
    }
  }
  for (auto& child : children) {
    json["properties"]["children"].push_back(eval(child.getObject()));
  }
  return json;
}
