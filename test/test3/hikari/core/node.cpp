#include <hikari/core/node.h>

auto hikari::core::NodeObject::getJSONString() const -> Str 
{
  Str res = "";
  res += "\{";
  res += " \"name\" : \"" + m_name + "\" ,";
  res += " \"type\" : \"Node\" ,";
  res += " \"properties\" : {";
  res += " \"children\": [";
  {
    auto children = getChildren();
    for (auto i = 0; i < children.size(); ++i) {
      res += " " + children[i]->getJSONString();
      if (i != children.size() - 1) { res += " ,"; }
    }
  }
  res += "],";
  res += " \"components\": [";
  {
    auto children = getComponents("NodeComponent");
    for (auto i = 0; i < children.size(); ++i) {
      res += " " + children[i]->getJSONString();
      if (i != children.size() - 1) { res += " ,"; }
    }
  }
  res += "],";
  {
    auto prop = Object::getProperty("local_transform");
    auto str = prop.getJSONString();
    res += "\"transform\" : " + str;
  }
  res += " }";
  res += "\}";
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
  auto name      = pb.getValue("name");
  if (name.getTypeIndex() == PropertyTypeIndex<Str>::value) {
    setName(*name.getValue<Str>());
  }

  auto parent    = pb.getValue("parent");
  if (parent.getTypeIndex() == PropertyTypeIndex<std::shared_ptr<Object>>::value) {
    auto parent_object = parent.getValue<std::shared_ptr<Object>>();
    if (parent_object->getTypeString() == "Node") {
      auto parent_node = std::static_pointer_cast<NodeObject>(parent_object);
      setParent(parent_node);
    }
    else {
      setParent({});
    }
  }

  auto children  = pb.getValue("children");
  if (children.getTypeIndex() == PropertyTypeIndex<std::vector<std::shared_ptr<Object>>>::value){
    popChildren();
    auto child_objects = children.getValue<std::vector<std::shared_ptr<Object>>>();
    for (auto& child_object : child_objects) {
      if (child_object->getTypeString() == "Node") {
        auto child_node = std::static_pointer_cast<NodeObject>(child_object);
        child_node->setParent(shared_from_this());
        continue;
      }
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
  auto children = std::vector<std::shared_ptr<Object>>();
  children.reserve(m_children.size());
  std::transform(m_children.begin(), m_children.end(), std::back_inserter(children), [](const auto& p) { return std::static_pointer_cast<Object>(p); });
  pb.setValue("name"     , getName());
  pb.setValue("parent"   , std::static_pointer_cast<Object>(getParent()));
  pb.setValue("children" , children);
  pb.setValue("transform", m_local_transform);
}

hikari::Bool hikari::core::NodeObject::hasProperty(const Str& name) const
{
  if (name == "name")            { return true; }
  if (name == "parent")          { return true; }
  if (name == "children")        { return true; }
  if (name == "child_count")     { return true; }
  if (name == "global_transform"){ return true; }
  if (name == "global_matrix")   { return true; }
  if (name == "global_position") { return true; }
  if (name == "global_rotation") { return true; }
  if (name == "global_scale")    { return true; }
  if (name == "local_transform") { return true; }
  if (name == "local_matirx")    { return true; }
  if (name == "local_position")  { return true; }
  if (name == "local_rotation")  { return true; }
  if (name == "local_scale")     { return true; }
  return false;
}

hikari::Bool hikari::core::NodeObject::setProperty(const Str& name, const Property& value)
{
  if (name == "name"      ) {
    auto str = value.getValue<Str>();
    if (str) {
      setName(*str);
      return true;
    }
    return false;
  }
  if (name == "parent"    ) {
    auto parent = value.getValue<std::shared_ptr<Object>>();
    if (parent) {
      if (!parent) {
        setParent(nullptr);
        return true;
      }
      else {
        if (parent->getTypeString() == "Node") {
          setParent(std::static_pointer_cast<NodeObject>(parent));
          return true;
        }
      }
    }
    return false;
  }
  if (name == "child_count") {
    auto childCount = value.getInteger();
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
    auto global_matrix = value.getMatrix();
    if (global_matrix) { setGlobalTransform(Transform(*global_matrix)); return true; }
    return false;
  }
  if (name == "local_matirx") {
    auto local_matrix = value.getMatrix();
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
    value = std::static_pointer_cast<Object>(m_parent.lock());
    return true;
  }
  if (name == "children") {
    auto children = getChildren();
    auto res = std::vector<std::shared_ptr<Object>>();
    res.reserve(children.size());
    std::copy(std::begin(children), std::end(children), std::back_inserter(res));
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
  auto res = std::vector<std::shared_ptr<NodeComponentObject>>();
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

 auto hikari::core::Node::operator[](const Str& name) -> PropertyRef
 {
   auto object = std::static_pointer_cast<ObjectType>(getObject());  return PropertyRef(object, name);
 }

 auto hikari::core::Node::operator[](const Str& name) const -> Property
 {
   auto object = std::static_pointer_cast<ObjectType>(getObject());
   if (object) {
     Property prop;
     object->getProperty(name,prop);
     return prop;
   }
   return Property();
 }

 auto hikari::core::Node::operator[](size_t idx) const -> Node
 {
   return Node(getChild(idx));
 }

 auto hikari::core::Node::operator[](size_t idx) -> NodeRef
 {
   auto object = getObject();
   return NodeRef(object,idx);
 }

 void hikari::core::Node::setPropertyBlock(const PropertyBlock& pb)
 {
   auto object = getObject();
   if (object) { object->setPropertyBlock(pb); }
 }

 void hikari::core::Node::getPropertyBlock(PropertyBlock& pb) const
 {
   auto object = getObject();
   if (object) { object->getPropertyBlock(pb); }
 }

auto hikari::core::Node::getJSONString() const -> std::string {
   auto object = getObject();
   if (object) { return object->getJSONString(); }
   else { return "null"; }
 }

 auto hikari::core::Node::getSize() const -> size_t
 {
   return getChildCount();
 }

 void hikari::core::Node::setSize(size_t count)
 {
   setChildCount(count);
 }

 auto hikari::core::Node::getName() const -> Str
 {
   auto object = getObject();
   if (object) {
     return object->getName();
   }
   else {
     return "";
   }
 }

 void hikari::core::Node::setName(const Str& name)
 {
   auto object = getObject();
   if (object) {
     return object->setName(name);
   }
 }

 auto hikari::core::Node::getObject() const -> std::shared_ptr<ObjectType>
 {
   return m_object;
 }

 auto hikari::core::Node::getChildCount() const -> size_t
 {
   if (m_object) {
     return m_object->getChildCount();
   }
   else {
     return 0;
   }
 }

 void hikari::core::Node::setChildCount(size_t count)
 {
   if (m_object) {
     return m_object->setChildCount(count);
   }
 }

 auto hikari::core::Node::getChildren() const -> std::vector<Node>
 {
   if (m_object) {
     auto children = m_object->getChildren();
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
   if (m_object) {
     auto tmp = std::vector<std::shared_ptr<NodeObject>>();
     for (auto& child : children) {
       tmp.push_back(child.getObject());
     }
     m_object->setChildren(tmp);
   }
 }

 void hikari::core::Node::popChildren()
 {
   if (m_object) {
     m_object->popChildren();
   }
   
 }

 auto hikari::core::Node::getChild(size_t idx) const -> Node
 {
   if (!m_object) { return Node(); }
   return Node(m_object->getChild(idx));
 }

 void hikari::core::Node::setChild(size_t idx, const Node& child)
 {
   if (!m_object) { return; }
   m_object->setChild(idx, child.getObject());
 }

 void hikari::core::Node::addChild(const Node& child)
 {
   if (!m_object) { return; }
   m_object->addChild(child.getObject());
 }

 void hikari::core::Node::popChild(size_t idx)
 {
   if (!m_object) { return; }
   m_object->popChild(idx);
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


 void hikari::core::NodeRef::operator=(const Node& node) noexcept
 {
   auto obj = m_object.lock();
   if (obj) {
     obj->setChild(m_idx, node.getObject());
   }
 }

 auto hikari::core::NodeRef::operator[](const Str& name) -> PropertyRef
 {
   auto object = getObject();
   return PropertyRef(object,name);
 }

 auto hikari::core::NodeRef::operator[](const Str& name) const -> Property
 {
   auto object = getObject();
   if (!object) { return Property(); }
   Property res;
   object->getProperty(name, res);
   return res;
 }

 auto hikari::core::NodeRef::operator[](size_t idx) const -> Node
 {
   return getChild(idx);
 }

 auto hikari::core::NodeRef::operator[](size_t idx) -> Ref
 {
   return NodeRef(getObject(), idx);
 }

 void hikari::core::NodeRef::setPropertyBlock(const PropertyBlock& pb)
 {
   auto object = getObject();
   if (object) { object->setPropertyBlock(pb); }
 }

 void hikari::core::NodeRef::getPropertyBlock(PropertyBlock& pb) const
 {
   auto object = getObject();
   if (object) { object->getPropertyBlock(pb); }
 }

 auto hikari::core::NodeRef::getSize() const -> size_t
 {
   return getChildCount();
 }

 void hikari::core::NodeRef::setSize(size_t count)
 {
   setChildCount(count);
 }

 auto hikari::core::NodeRef::getName() const -> Str
 {
   auto object = getObject();
   if (object) {
     return object->getName();
   }
   else {
     return "";
   }
 }

 void hikari::core::NodeRef::setName(const Str& name)
 {
   auto object = getObject();
   if (object) {
     return object->setName(name);
   }
 }

 auto hikari::core::NodeRef::getObject() const -> std::shared_ptr<ObjectType>
 {
   auto object = m_object.lock();
   if (!object) { return nullptr; }
   return object->getChild(m_idx);
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
   return NodeRef(object,idx);
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

 auto hikari::core::convertToJSONString(const core::Node& v) -> Str
 {
   return v.getJSONString();
 }

 auto hikari::core::convertToString(const core::Node& v) -> Str
 {
   return v.getJSONString();
 }

 auto hikari::core::convertStringToNode(const Str& str) -> Node
 {
   return convertStringToNode(convertStringToJSON(str));
 }


 auto hikari::core::convertJSONToNode(const Json& json) -> Node
 {
   if (json.is_object()) {
     auto iter_type = json.find("type");
     auto iter_name = json.find("name");
     auto iter_properties = json.find("properties");
     if (iter_type == json.end()) { return Node(nullptr); }
     if (iter_name == json.end()) { return Node(nullptr); }
     if (iter_properties == json.end()) { return Node(nullptr); }
     if (!iter_type.value().is_string()) { return Node(nullptr); }
     if (!iter_name.value().is_string()) { return Node(nullptr); }
     if (!iter_properties.value().is_object()) { return Node(nullptr); }
     if (iter_type.value().get<std::string>() != "Node") { return Node(nullptr); }
     auto& properties = iter_properties.value();
     auto iter_children = properties.find("children");
     if (iter_children == properties.end()) { return Node(nullptr); }
     if (!iter_children.value().is_array()) { return Node(nullptr); }
     auto node = Node(iter_name.value().get<std::string>());
     auto iter_transform = properties.find("transform");
     if (iter_transform != properties.end()) {
       auto prop = convertJSONStringToTransform(iter_transform.value().dump());
       if (prop) {
         node.setLocalTransform(*prop);
       }
     }
     for (auto& child : iter_children.value()) { node.addChild(convertJSONToNode(child)); }
     return node;
   }
   else {
     return Node(nullptr);
   }
 }

 auto hikari::core::convertNodeToJSON(const Node& node) -> Json
 {
   return Json();
 }

 hikari::core::NodeComponentObject::~NodeComponentObject() noexcept {}

 auto hikari::core::NodeComponentObject::getName() const -> Str { auto node = getNode(); if (node) { return node->getName(); }  return ""; }

 auto hikari::core::NodeComponentObject::getNode() const -> std::shared_ptr<NodeObject> { return m_node.lock(); }

 hikari::core::NodeComponentObject::NodeComponentObject(const std::shared_ptr<NodeObject>& node)
   :m_node{node}
 {
 }

 auto hikari::core::NodeComponent::operator[](const Str& name) -> PropertyRef
 {
   return PropertyRef(std::static_pointer_cast<hikari::core::Object>(m_object.lock()),name);
 }

 auto hikari::core::NodeComponent::operator[](const Str& name) const -> Property { return getValue(name); }
