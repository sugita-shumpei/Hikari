#include "serialized_data.h"
#include <scn/scn.h>
#include <fstream>

auto hikari::assets::mitsuba::spectrum::SerializedImporter::create(const std::string& filename) -> std::shared_ptr<SerializedImporter>
{
  return std::shared_ptr<SerializedImporter>(new SerializedImporter(filename));
}

auto hikari::assets::mitsuba::spectrum::SerializedImporter::load() const -> std::shared_ptr<Serialized>
{
  auto file = std::ifstream(m_filename);
  if (file.fail()) { return nullptr; }
  auto res = std::shared_ptr<Serialized>(new Serialized());
  auto tmp = std::string();
  while (std::getline(file, tmp)) {
    float tmp1, tmp2;
    if (scn::scan(tmp, "{} {}", tmp1, tmp2)) {
      res->wavelengths.push_back(tmp1);
      res->weights.push_back(tmp2);
    }
  }
  file.close();
  return res;
}
