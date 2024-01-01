#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <hikari/assets/image/importer.h>
#include <stb_image_write.h>

int main() {
  auto& image_importer = hikari::ImageImporter::getInstance();
  if (image_importer.load(R"(D:\Users\shums\Documents\C++\Hikari\data\mitsuba\classroom\textures\spaichingen_hill_2k.exr)")) {
      auto image = image_importer.get("spaichingen_hill_2k.exr");
      auto bitmap = image->getImage(0);
      stbi_write_hdr("spaichingen_hill_2k.hdr", image->getWidth(), image->getHeight(), image->getChannel(), (const float*)bitmap->getData());
      image_importer.free(R"(spaichingen_hill_2k.exr)");
      return 0;
  }
  return -1;
}
