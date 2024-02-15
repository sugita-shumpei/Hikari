#include "irregular.h"

auto hikari::spectrum::SpectrumIrregularObject::getPropertyNames() const -> std::vector<Str> 
{
  return std::vector<Str>{
    //"min_wavelength",
    //"max_wavelength"
    "wavelengths",
    "intensities"
    //"values"
  };
}

void hikari::spectrum::SpectrumIrregularObject::getPropertyBlock(PropertyBlockBase<Object>& pb) const
{
  pb.clear();
  pb.setValue("wavelengths", getWaveLengths());
  pb.setValue("intensities", getIntensities());
}

void hikari::spectrum::SpectrumIrregularObject::setPropertyBlock(const PropertyBlockBase<Object>& pb)
{
  auto wavelengths = pb.getValue<Array<F32>>("wavelengths");
  auto intensities = pb.getValue<Array<F32>>("intensities");
  if (wavelengths.size() == intensities.size()) {
    if (wavelengths.size() < 2) { return; }
    auto res = std::vector<Pair<F32, F32>>();
    res.reserve(wavelengths.size());
    for (size_t i = 0; i < res.capacity(); ++i) {
      res.push_back({ (wavelengths)[i],(intensities)[i] });
    }
    setWaveLengthsAndIntensities(res);
  }
}

bool hikari::spectrum::SpectrumIrregularObject::hasProperty(const Str& name) const
{
  if (name == "wavelengths"     ) { return true; }
  if (name == "intensities"     ) { return true; }
  if (name == "wavelengths.size") { return true; }
  if (name == "intensities.size") { return true; }
  if (name == "values"          ) { return true; }
  if (name == "values.size"     ) { return true; }
  if (name == "min_wavelength"  ) { return true; }
  if (name == "max_wavelength"  ) { return true; }
  return false;
}

bool hikari::spectrum::SpectrumIrregularObject::getProperty(const Str& name, PropertyBase<Object>& prop) const
{
  if (name == "wavelengths"     ) { prop.setValue(getWaveLengths());return true; }
  if (name == "intensities"     ) { prop.setValue(getIntensities()); return true; }
  if (name == "wavelengths.size") { prop.setValue(getSize()); return true; }
  if (name == "intensities.size") { prop.setValue(getSize()); return true; }
  if (name == "values"          ) {
    auto res = std::vector<Vec2>();
    auto tmp = getWaveLengthsAndIntensities();
    for (auto& [len, inte] : tmp) {
      res.push_back(Vec2(len, inte));
    }
    prop.setValue(res);
    return true;
  }
  if (name == "values.size"     ) { prop.setValue(getSize()); return true; }
  if (name == "min_wavelength"  ) { prop.setValue(getMinWaveLength()); return true; }
  if (name == "max_wavelength"  ) { prop.setValue(getMaxWaveLength()); return true; }
  return false;
}

bool hikari::spectrum::SpectrumIrregularObject::setProperty(const Str& name, const PropertyBase<Object>& prop)
{
  if (name == "values") {
    auto tmp = prop.getValue<Array<Vec2>>();
    if (tmp.size() < 2) { return false; }
    auto res = std::vector<Pair<F32, F32>>();
    for (auto& v : tmp) {
      res.push_back({v.x,v.y});
    }
    setWaveLengthsAndIntensities(res);
    return true;
  }
  return false;
}

auto hikari::spectrum::SpectrumIrregularObject::sample(F32 wavelength) const -> F32 
{
  auto min_elem = m_wave_lengths_and_intensities.front();
  auto max_elem = m_wave_lengths_and_intensities.back();
  if (min_elem.first == wavelength) { return min_elem.second; }
  if (max_elem.first == wavelength) { return max_elem.second; }
  if (min_elem.first > wavelength) { return 0.0f; }
  if (max_elem.first < wavelength) { return 0.0f; }
  if (m_wave_lengths_and_intensities.size() == 2) {
    auto rate = (wavelength - min_elem.first) / (max_elem.first - min_elem.first);
    return (1.0f - rate) * min_elem.second + rate * max_elem.second;
  }
  auto iter = std::lower_bound(
    m_wave_lengths_and_intensities.begin(),
    m_wave_lengths_and_intensities.end(), Pair<F32,F32>{ wavelength,0.0f },
    [](const auto& lhs, const auto& rhs) {return lhs.first < rhs.first; }
  );
  if (iter->first == wavelength) { return iter->second; }
  {
    auto beg_elem = iter - 1;
    auto end_elem = iter;
    auto rate = (wavelength - beg_elem->first) / (end_elem->first - beg_elem->first);
    return (1.0f - rate) * beg_elem->second + rate * end_elem->second;

  }
}

auto hikari::spectrum::SpectrumIrregularObject::getXYZColor() const -> ColorXYZ 
{
  
  return ColorXYZ ();
}

auto hikari::spectrum::SpectrumIrregularObject::getMinWaveLength() const -> F32 
{
  return m_wave_lengths_and_intensities.front().first;
}

auto hikari::spectrum::SpectrumIrregularObject::getMaxWaveLength() const -> F32 
{
  return m_wave_lengths_and_intensities.back().first;
}

auto hikari::spectrum::SpectrumIrregularObject::getWaveLengthsAndIntensities() const -> Array<Pair<F32, F32>>
{
  return m_wave_lengths_and_intensities;
}

void hikari::spectrum::SpectrumIrregularObject::setWaveLengthsAndIntensities(const Array<Pair<F32, F32>>& wavelength_and_intensities)
{
  if (wavelength_and_intensities.size() >= 2) {
    m_wave_lengths_and_intensities = wavelength_and_intensities;
    std::sort(std::begin(m_wave_lengths_and_intensities), std::end(m_wave_lengths_and_intensities),
      [](const auto& l, const auto& r) { return l.first < r.first; }
    );
  }
}

auto hikari::spectrum::SpectrumIrregularObject::getWaveLengths() const -> Array<F32>
{
  std::vector<F32> res; res.reserve(m_wave_lengths_and_intensities.size());
  for (auto& [len, inten] : m_wave_lengths_and_intensities) {
    res.push_back(len);
  }
  return res;
}

auto hikari::spectrum::SpectrumIrregularObject::getIntensities() const -> Array<F32>
{
  std::vector<F32> res; res.reserve(m_wave_lengths_and_intensities.size());
  for (auto& [len, inten] : m_wave_lengths_and_intensities) {
    res.push_back(inten);
  }
  return res;
}

auto hikari::spectrum::SpectrumIrregularObject::getIntensity(F32 wavelength) const -> F32
{
  auto iter = std::find_if(
    m_wave_lengths_and_intensities.begin(),
    m_wave_lengths_and_intensities.end(),
    [wavelength](const auto& pair) {
      return pair.first == wavelength;
    }
  );
  if (iter != std::end(m_wave_lengths_and_intensities)) {
    return iter->second;
  }
  else {
    return 0.0f;
  }
}

void hikari::spectrum::SpectrumIrregularObject::setIntensity(F32 wavelength, F32 intensity)
{
  if (m_wave_lengths_and_intensities.front().first == wavelength) {
    m_wave_lengths_and_intensities.front().second = intensity;
    return;
  }
  if (m_wave_lengths_and_intensities.back().first  == wavelength) {
    m_wave_lengths_and_intensities.back().second = intensity;
    return;
  }
  if (m_wave_lengths_and_intensities.front().first  > wavelength) {
    auto capacity = m_wave_lengths_and_intensities.capacity();
    auto size = m_wave_lengths_and_intensities.size();
    if (size + 1 > capacity) {
      m_wave_lengths_and_intensities.reserve(size + 1);
    }
    m_wave_lengths_and_intensities.insert(m_wave_lengths_and_intensities.begin(), { wavelength,intensity });
    return;
  }
  if (m_wave_lengths_and_intensities.back().first   < wavelength) {
    m_wave_lengths_and_intensities.push_back({ wavelength,intensity });
    return;
  }
  if (m_wave_lengths_and_intensities.size() == 2) {
    auto capacity = m_wave_lengths_and_intensities.capacity();
    auto size = m_wave_lengths_and_intensities.size();
    if (size + 1 > capacity) {
      m_wave_lengths_and_intensities.reserve(size + 1);
    }
    m_wave_lengths_and_intensities.insert(m_wave_lengths_and_intensities.begin()+1, { wavelength,intensity });
    return;
  }
  auto iter = std::lower_bound(
    m_wave_lengths_and_intensities.begin(),
    m_wave_lengths_and_intensities.end(), Pair<F32,F32>{wavelength,0.0f},
    [](const auto& lhs, const auto& rhs) {return lhs.first < rhs.first; }
  );
  if (iter->first == wavelength) {
    iter->second = intensity;
    return;
  }
  else {
    auto distance = std::distance(std::begin(m_wave_lengths_and_intensities), iter);
    auto capacity = m_wave_lengths_and_intensities.capacity();
    auto size = m_wave_lengths_and_intensities.size();
    if (size + 1 > capacity) {
      m_wave_lengths_and_intensities.reserve(size + 1);
      iter = std::begin(m_wave_lengths_and_intensities) + distance;
    }
    m_wave_lengths_and_intensities.insert(iter, { wavelength,intensity });
    return;
  }

}

void hikari::spectrum::SpectrumIrregularObject::popIntensity(F32 wavelength)
{
  // 長さは最低でも2以上でないといけない
  if (m_wave_lengths_and_intensities.size() == 2) { return; }
  auto iter = std::find_if(
    m_wave_lengths_and_intensities.begin(),
    m_wave_lengths_and_intensities.end(),
    [wavelength](const auto& pair) { return pair.first == wavelength; }
  );
  if (iter != std::end(m_wave_lengths_and_intensities)) { m_wave_lengths_and_intensities.erase(iter); }
}

bool hikari::spectrum::SpectrumIrregularObject::hasIntensity(F32 wavelength) const noexcept
{
 return std::find_if(
    m_wave_lengths_and_intensities.begin(),
    m_wave_lengths_and_intensities.end(),
    [wavelength](const auto& pair) { return pair.first == wavelength; }
  ) != std::end(m_wave_lengths_and_intensities);
}

auto hikari::spectrum::SpectrumIrregularObject::getSize() const -> U32
{
  return m_wave_lengths_and_intensities.size();
}

auto hikari::spectrum::SpectrumIrregularSerializer::getTypeString() const noexcept -> Str 
{
  return SpectrumIrregularObject::TypeString();
}

auto hikari::spectrum::SpectrumIrregularSerializer::eval(const std::shared_ptr<Object>& object) const -> Json 
{
  auto regular = ObjectUtils::convert<SpectrumIrregularObject>(object);
  if (!regular) { return Json(); }
  Json json = {};
  json["type"] = "SpectrumIrregular";
  json["properties"] = Json();
  json["properties"]["wavelengths"] = regular->getWaveLengths();
  json["properties"]["intensities"] = regular->getIntensities();
  return json;
}

auto hikari::spectrum::SpectrumIrregularDeserializer::getTypeString() const noexcept -> Str 
{
  return SpectrumIrregularObject::TypeString();
}

auto hikari::spectrum::SpectrumIrregularDeserializer::eval(const Json& json) const -> std::shared_ptr<Object> 
{
  auto properties = json.find("properties");
  if (properties == json.end()) { return nullptr; }
  auto irregular = SpectrumIrregularObject::create();
  if (properties.value().is_null()) { return irregular; }
  if (!properties.value().is_object()) { return nullptr; }
  auto wavelengths = properties.value().find("wavelengths");
  auto intensities = properties.value().find("intensities");
  if (intensities != properties.value().end() &&
      wavelengths != properties.value().end()) {
    try {
      auto val_intensities = intensities.value().get<Array<F32>>();
      auto val_wavelengths = wavelengths.value().get<Array<F32>>();
      if (val_intensities.size() != val_wavelengths.size()) {
        return nullptr;
      }
      if (val_intensities.size() < 2) { return nullptr; }
      Array<Pair<F32, F32>> res = {};
      res.reserve(val_intensities.size());
      for (size_t i = 0; i < val_intensities.capacity(); ++i) {
        res.push_back(Pair<F32,F32>{ val_wavelengths[i],val_intensities[i] });
      }
      irregular->setWaveLengthsAndIntensities(res);
    }
    catch (...) {
      return nullptr;
    }
  }
  return irregular;
}
