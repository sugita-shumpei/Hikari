#pragma once
#include <memory>
#include <string>
#include <hikari/core/mipmap.h>
namespace hikari {
  struct ImageImporter {
    static auto getInstance() -> ImageImporter&;
    virtual ~ImageImporter();

    ImageImporter(const ImageImporter&) = delete;
    ImageImporter(ImageImporter&&) = delete;
    ImageImporter& operator=(const ImageImporter&) = delete;
    ImageImporter& operator=(ImageImporter&&) = delete;

    auto load(const String& filename)->MipmapPtr;
    void free(const String& filename);

    auto get(const String & filename) const->MipmapPtr;
    void clear();
  private:
    auto loadStbImage(const String& filename) -> MipmapPtr;
    auto loadHdrImage(const String& filename) -> MipmapPtr;
    auto loadExrImage(const String& filename) -> MipmapPtr;
    auto loadPfmImage(const String& filename) -> MipmapPtr;
  private:
    ImageImporter() {}
    std::unordered_map<String, MipmapPtr> m_mipmaps = {};
  };
}
