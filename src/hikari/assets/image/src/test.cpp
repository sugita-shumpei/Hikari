#include <hikari/assets/image/importer.h>
#include <hikari/assets/image/exporter.h>
int main() {
  auto& image_importer = hikari::ImageImporter::getInstance();
  if (image_importer.load(R"(D:\Users\shums\Documents\CMake\RTLib\Data\Textures\evening_road_01_4k.hdr)")) {
      auto image = image_importer.get("evening_road_01_4k.hdr");
      auto bitmap = image->getImage(0);
      hikari::ImageExporter::save("evening_road_01_4k.hdr", image);
      hikari::ImageExporter::save("evening_road_01_4k.exr", image);
      image_importer.free(R"(evening_road_01_4k.hdr)");
      return 0;
  }
  return -1;
}
