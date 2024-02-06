#include "transform.h"
#include <fmt/format.h>
#include <sstream>

auto hikari::assets::mitsuba::XMLTransform::create(const XMLContextPtr& context) -> std::shared_ptr<XMLTransform>
{
  return std::shared_ptr<XMLTransform>(new XMLTransform(context));
}

auto hikari::assets::mitsuba::XMLTransform::toStrings() const -> std::vector<std::string> 
{
  std::vector<std::string> res = {};
  res.push_back("\"object_type\":\"transform\"");
  std::string element_str = "";
  element_str += "\"elements\": [\n";
  auto idx = 0;
  for (auto& elem : m_elements) {
    std::stringstream ss;
    ss << elem->toString();
    std::string tmp;
    while (std::getline(ss, tmp, '\n')) {
      element_str += "  " + tmp;
      if (ss.rdbuf()->in_avail()) { element_str += "\n"; }
    }
    if (idx == m_elements.size() - 1) {
      element_str += "  \n";
    }
    else {
      element_str += "  ,\n";
    }
    idx++;
  }
  element_str += "]";
  res.push_back(element_str);
  return res;
}

void hikari::assets::mitsuba::XMLTransform::addElement(const ElementPtr& elem) noexcept
{
  m_elements.push_back(elem);
}

void hikari::assets::mitsuba::XMLTransform::setElement(size_t idx, const ElementPtr& elem) noexcept
{
  if (idx >= getSize()) {
    setSize(idx + 1);
  }
  m_elements[idx] = elem;
}

auto hikari::assets::mitsuba::XMLTransform::getElement(size_t idx) const noexcept -> ElementPtr
{
  if (m_elements.size() > idx) { return m_elements[idx]; }
  else { return nullptr; }
}

auto hikari::assets::mitsuba::XMLTransform::getSize() const noexcept -> size_t
{
  return m_elements.size();
}

void hikari::assets::mitsuba::XMLTransform::setSize(size_t count) noexcept
{
  m_elements.resize(count, nullptr);
  m_elements.shrink_to_fit();
}

auto hikari::assets::mitsuba::XMLTransformElementTranslate::create(const XMLContextPtr& context, const glm::vec3& position) -> std::shared_ptr<XMLTransformElementTranslate>
{
  return std::shared_ptr<XMLTransformElementTranslate>(new XMLTransformElementTranslate(context,position));
}

hikari::assets::mitsuba::XMLTransformElementTranslate::~XMLTransformElementTranslate() noexcept {}

auto hikari::assets::mitsuba::XMLTransformElementTranslate::toString() const -> std::string 
{
  auto res = std::string("");
  res += "{\n";
  res += "  \"type\":\"translate\",\n";
  res += "  \"value\" :" + fmt::format("[{},{},{}]", m_value.x, m_value.y, m_value.z) + "\n";
  res += "}";
  return res;
}

void hikari::assets::mitsuba::XMLTransformElementTranslate::setValue(const glm::vec3& position) noexcept
{
  m_value = position;
}

auto hikari::assets::mitsuba::XMLTransformElementTranslate::getValue() const noexcept -> glm::vec3
{
  return m_value;
}

hikari::assets::mitsuba::XMLTransformElementTranslate::XMLTransformElementTranslate(const XMLContextPtr& context, const glm::vec3& position)
  :XMLTransformElement(Type::eTranslate),m_context{context},m_value{position}
{
}

auto hikari::assets::mitsuba::XMLTransformElementRotation::create(const XMLContextPtr& context, const glm::vec3& axis, float angle) -> std::shared_ptr<XMLTransformElementRotation>
{
  return std::shared_ptr<XMLTransformElementRotation>(new XMLTransformElementRotation(context,axis,angle));
}

hikari::assets::mitsuba::XMLTransformElementRotation::~XMLTransformElementRotation() noexcept
{
}


auto hikari::assets::mitsuba::XMLTransformElementRotation::toString() const -> std::string 
{
  auto res = std::string("");
  res += "{\n";
  res += "  \"type\":\"rotation\",\n";
  res += "  \"value\" :" + fmt::format("[{},{},{}]", m_value.x, m_value.y, m_value.z) + ",\n";
  res += "  \"angle\" :" + fmt::format("{}", m_angle) + "\n";
  res += "}";
  return res;
}

void hikari::assets::mitsuba::XMLTransformElementRotation::setValue(const glm::vec3& axis) noexcept
{
  m_value = axis;
}

auto hikari::assets::mitsuba::XMLTransformElementRotation::getValue() const noexcept -> glm::vec3
{
  return m_value;
}

void hikari::assets::mitsuba::XMLTransformElementRotation::setAngle(float angle) noexcept
{
  m_angle = angle;
}

auto hikari::assets::mitsuba::XMLTransformElementRotation::getAngle() const noexcept -> float
{
  return m_angle;
}

hikari::assets::mitsuba::XMLTransformElementRotation::XMLTransformElementRotation(const XMLContextPtr& context, const glm::vec3& axis, float angle)
  :XMLTransformElement(Type::eRotate), m_context{ context }, m_value{axis},m_angle{angle}
{
}

auto hikari::assets::mitsuba::XMLTransformElementScale::create(const XMLContextPtr& context, const glm::vec3& value) -> std::shared_ptr<XMLTransformElementScale>
{
  return std::shared_ptr<XMLTransformElementScale>(new XMLTransformElementScale(context,value));
}

hikari::assets::mitsuba::XMLTransformElementScale::~XMLTransformElementScale() noexcept
{
}

auto hikari::assets::mitsuba::XMLTransformElementScale::toString() const -> std::string 
{
  auto res = std::string("");
  res += "{\n";
  res += "  \"type\":\"scale\",\n";
  res += "  \"value\" :" + fmt::format("[{},{},{}]", m_value.x, m_value.y, m_value.z) + "\n";
  res += "}";
  return res;
}

void hikari::assets::mitsuba::XMLTransformElementScale::setValue(const glm::vec3& value) noexcept
{
  m_value = value;
}

auto hikari::assets::mitsuba::XMLTransformElementScale::getValue() const noexcept -> glm::vec3
{
  return m_value;
}

hikari::assets::mitsuba::XMLTransformElementScale::XMLTransformElementScale(const XMLContextPtr& context, const glm::vec3& value):
  XMLTransformElement(Type::eScale),m_context{context},m_value{value}
{
}

auto hikari::assets::mitsuba::XMLTransformElementMatrix::create(const XMLContextPtr& context, const glm::mat4& value) -> std::shared_ptr<XMLTransformElementMatrix>
{
  return std::shared_ptr<XMLTransformElementMatrix>(new XMLTransformElementMatrix(context,value));
}

hikari::assets::mitsuba::XMLTransformElementMatrix::~XMLTransformElementMatrix() noexcept
{
}


auto hikari::assets::mitsuba::XMLTransformElementMatrix::toString() const -> std::string
{
  auto res = std::string("");
  res += "{\n";
  res += "  \"type\":\"matrix\",\n";
  res += "  \"value\" : \n";
  res += "  [\n";
  res += "    " + fmt::format("{},{},{},{},\n", m_value[0][0], m_value[1][0], m_value[2][0], m_value[3][0]);
  res += "    " + fmt::format("{},{},{},{},\n", m_value[0][1], m_value[1][1], m_value[2][1], m_value[3][1]);
  res += "    " + fmt::format("{},{},{},{},\n", m_value[0][2], m_value[1][2], m_value[2][2], m_value[3][2]);
  res += "    " + fmt::format("{},{},{},{}\n", m_value[0][3], m_value[1][3], m_value[2][3], m_value[3][3]);
  res += "  ]\n";
  res += "}";
  return res;
}

void hikari::assets::mitsuba::XMLTransformElementMatrix::setValue(const glm::mat4& value) noexcept
{
  m_value = value;
}

auto hikari::assets::mitsuba::XMLTransformElementMatrix::getValue() const noexcept -> glm::mat4
{
  return m_value;
}

hikari::assets::mitsuba::XMLTransformElementMatrix::XMLTransformElementMatrix(const XMLContextPtr& context, const glm::mat4& value)
  :XMLTransformElement(Type::eMatrix),m_context{context},m_value{value}
{
}

auto hikari::assets::mitsuba::XMLTransformElementLookAt::create(const XMLContextPtr& context, const glm::vec3& origin, const glm::vec3& target, const glm::vec3& up) -> std::shared_ptr<XMLTransformElementLookAt>
{
  return std::shared_ptr<XMLTransformElementLookAt>(new XMLTransformElementLookAt(context,origin,target,up));
}

hikari::assets::mitsuba::XMLTransformElementLookAt::~XMLTransformElementLookAt() noexcept
{
}

auto hikari::assets::mitsuba::XMLTransformElementLookAt::toString() const -> std::string 
{
  auto res = std::string("");
  res += "{\n";
  res += "  \"type\":\"lookat\",\n";
  res += "  \"origin\":" + fmt::format("[{},{},{}]", m_origin.x,m_origin.y,m_origin.z) + ",\n";
  res += "  \"target\":" + fmt::format("[{},{},{}]", m_target.x,m_target.y,m_target.z) + ",\n";
  res += "  \"up\"    :" + fmt::format("[{},{},{}]", m_up.x    ,m_up.y    ,m_up.z    ) + "\n";
  res += "}";
  return res;
}

void hikari::assets::mitsuba::XMLTransformElementLookAt::setOrigin(const glm::vec3& origin) noexcept
{
  m_origin = origin;
}

void hikari::assets::mitsuba::XMLTransformElementLookAt::setTarget(const glm::vec3& target) noexcept
{
  m_target = target;
}

void hikari::assets::mitsuba::XMLTransformElementLookAt::setUp(const glm::vec3& up) noexcept
{
  m_up = up;
}

auto hikari::assets::mitsuba::XMLTransformElementLookAt::getOrigin() const noexcept -> glm::vec3
{
  return m_origin;
}

auto hikari::assets::mitsuba::XMLTransformElementLookAt::getTarget() const noexcept -> glm::vec3
{
  return m_target;
}

auto hikari::assets::mitsuba::XMLTransformElementLookAt::getUp() const noexcept -> glm::vec3
{
  return m_up;
}

hikari::assets::mitsuba::XMLTransformElementLookAt::XMLTransformElementLookAt(const XMLContextPtr& context, const glm::vec3& origin, const glm::vec3& target, const glm::vec3& up)
  :XMLTransformElement(Type::eLookAt),m_context{context},m_origin{origin},m_up{up},m_target{target}
{
}
