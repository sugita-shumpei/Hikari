#include "shape.h"

auto hikari::assets::mitsuba::XMLShape::create(const XMLContextPtr& context, const std::string& plugin_type, const std::string& id) -> std::shared_ptr<XMLShape>
{
  auto shape = std::shared_ptr<XMLShape>(new XMLShape(context,plugin_type,id));
  context->setRefObject(shape);
  return shape;
}

hikari::assets::mitsuba::XMLShape::~XMLShape() noexcept
{
}

auto hikari::assets::mitsuba::XMLShape::getBsdf() const noexcept -> std::shared_ptr<XMLBsdf>
{
  auto&  properties = getProperties();
  auto   res        = std::static_pointer_cast<XMLBsdf>(getNestObj(Type::eBsdf, 0));
  if (res) { return res; }
  auto   context    = getContext();
  for (size_t i   = 0; i < getNestRefCount(); ++i) {
    auto ref = getNestRef(i);
    auto obj = context->getObject(ref.id);
    if (obj) {
      if (obj->getObjectType()==Type::eBsdf){ return std::static_pointer_cast<XMLBsdf>(obj); }
    }
  }
  return nullptr;
}

void hikari::assets::mitsuba::XMLShape::setBsdf(const std::shared_ptr<XMLBsdf>& bsdf)noexcept
{
  auto& properties = getProperties();
  setNestObj(0, bsdf);
}

auto hikari::assets::mitsuba::XMLShape::getInteriorMedium() const noexcept -> std::shared_ptr<XMLMedium>
{
  auto& properties = getProperties();
  {
    auto object = properties.getObject("interior");
    if (object) {
      if (object->getObjectType() == Type::eMedium) { return std::static_pointer_cast<XMLMedium>(object); }
      else { return nullptr; }
    }
  }
  auto ref = properties.getRef("interior");
  if (!ref) { return nullptr; }
  auto context = getContext();
  auto object = context->getObject(ref->id);
  if (object->getObjectType() != Type::eMedium) { return nullptr; }
  return std::static_pointer_cast<XMLMedium>(object);
}

auto hikari::assets::mitsuba::XMLShape::getExteriorMedium() const noexcept -> std::shared_ptr<XMLMedium>
{
  auto& properties = getProperties();
  {
    auto object    = properties.getObject("exterior");
    if (object) {
      if (object->getObjectType() == Type::eMedium) { return std::static_pointer_cast<XMLMedium>(object); }
      else { return nullptr; }
    }
  }
  auto ref     = properties.getRef("exterior");
  if (!ref) { return nullptr; }
  auto context = getContext();
  auto object  = context->getObject(ref->id);
  if (object->getObjectType() != Type::eMedium) { return nullptr; }
  return std::static_pointer_cast<XMLMedium>(object);
}

void hikari::assets::mitsuba::XMLShape::setInteriorMedium(const std::shared_ptr<XMLMedium>& medium) noexcept
{
  if (!medium) {
    return;
  }

  auto& properties = getProperties();
  auto id = medium->getID();
  if (id == "") {
    properties.setValue("interior", medium);
  }
  else {
    properties.setValue("interior", XMLRef(id));
  }
}

void hikari::assets::mitsuba::XMLShape::setExteriorMedium(const std::shared_ptr<XMLMedium>& medium) noexcept
{
  if (!medium) {
    return;
  }
  auto& properties = getProperties();
  auto id = medium->getID();
  if (id == "") {
    properties.setValue("exterior", medium);
  }
  else {
    properties.setValue("exterior", XMLRef(id));
  }
}

auto hikari::assets::mitsuba::XMLShape::getContext() const -> XMLContextPtr
{
  return m_context.lock();
}

hikari::assets::mitsuba::XMLShape::XMLShape(const XMLContextPtr& context, const std::string& plugin_type, const std::string& ref_id) noexcept :
  XMLReferableObject(Type::eShape, plugin_type, ref_id),m_context{context}
{
}
