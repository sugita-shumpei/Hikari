#pragma once
#include <memory>
#include <string>
#include <hikari/core/mipmap.h>
namespace hikari {
  struct ImageExporter {
    static bool save(const String& filename, const MipmapPtr& mipmap);
  private:
    static bool saveStbImage(const String& filename, const MipmapPtr& mipmap);
    static bool saveExrImage(const String& filename, const MipmapPtr& mipmap);
    static bool saveHdrImage(const String& filename, const MipmapPtr& mipmap);
    static bool savePfmImage(const String& filename, const MipmapPtr& mipmap);
  };
}
