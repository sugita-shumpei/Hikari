#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include <hikari/assets/image/exporter.h>
#include <filesystem>
#include <stb_image_write.h>
#include <tinyexr.h>

bool hikari::ImageExporter::save(const String& filename, const MipmapPtr& mipmap)
{
  auto path = std::filesystem::path(filename);
  auto ext = path.extension().string();
  if ((ext == ".jpg") ||
      (ext == ".jpeg") ||
      (ext == ".png") ||
      (ext == ".tga") ||
      (ext == ".bmp")) {
    return saveStbImage(filename, mipmap);
  }
  if ((ext == ".exr")) {
    return saveExrImage(filename, mipmap);
  }
  if ((ext == ".hdr")) {
    return saveHdrImage(filename, mipmap);
  }
  if ((ext == ".pfm")) {
    return savePfmImage(filename, mipmap);
  }
  return false;
}

bool hikari::ImageExporter::saveStbImage(const String& filename, const MipmapPtr& mipmap)
{
  if (!mipmap) { return false; }
  if (mipmap->getDimension()!= hikari::MipmapDimension::e2D) { return false; }
  if (mipmap->getDataType() != hikari::MipmapDataType::eU8 ) { return false; }
  auto channel  = mipmap->getChannel();
  auto image    = mipmap->getImage(0);
  auto data_ptr = image->getData();

  auto path = std::filesystem::path(filename);
  auto ext = path.extension().string();
  if ((ext == ".jpg") || (ext == ".jpeg")) { stbi_write_jpg(filename.c_str(), image->getWidth(), image->getHeight(), channel, data_ptr, 0); return true; }
  if ((ext == ".png")) { stbi_write_png(filename.c_str(), image->getWidth(), image->getHeight(), channel, data_ptr, image->getWidth() * channel); return true; }
  if ((ext == ".tga")) { stbi_write_tga(filename.c_str(), image->getWidth(), image->getHeight(), channel, data_ptr);  return true; }
  if ((ext == ".bmp")) { stbi_write_bmp(filename.c_str(), image->getWidth(), image->getHeight(), channel, data_ptr);  return true; }
  return false;
}

bool hikari::ImageExporter::saveExrImage(const String& filename, const MipmapPtr& mipmap)
{
  if (!mipmap) { return false; }
  if (mipmap->getDimension() != hikari::MipmapDimension::e2D) { return false; }

  EXRHeader exr_header;
  InitEXRHeader(&exr_header);

  EXRImage  exr_image;
  InitEXRImage(&exr_image);

  auto channel  = mipmap->getChannel();
  auto image    = mipmap->getImage(0);
  auto data_ptr = image->getData();

  auto image_size = image->getWidth() * image->getHeight();

  const char* channel_names[4] = {"R","G","B","A"};

  if ((mipmap->getDataType() == hikari::MipmapDataType::eF16)) {
    std::vector<F16> pixels[4];
    for (size_t k = 0; k < channel; ++k) {
      pixels[k] = std::vector<F16>(image_size);
      for (size_t j = 0; j < image_size; ++j) {
        pixels[k][j] = ((const F16*)data_ptr)[channel*j+k];
      }
    }
    void* image_ptrs[4];
    for (size_t k = 0; k < channel; ++k) {
      image_ptrs[k] = pixels[channel - 1 - k].data();
    }
    exr_image.num_channels = channel;
    exr_image.images = (unsigned char**)image_ptrs;
    exr_image.width  = image->getWidth();
    exr_image.height = image->getHeight();

    std::vector< EXRChannelInfo> channels(channel, EXRChannelInfo{});

    exr_header.num_channels = channel;
    exr_header.channels     = channels.data();

    std::vector<int>         tmp_pixel_types(channel, 0);
    std::vector<int>         tmp_req_pixel_types(channel, 0);
    std::vector<const char*> tmp_channel_names(channel, "");
    for (size_t k = 0; k < channel; ++k) {// R G B A
      channels[channel - 1 - k].name[0] = channel_names[k][0];
      channels[channel - 1 - k].name[1] = '\0';
      tmp_pixel_types[k] = TINYEXR_PIXELTYPE_HALF;
      tmp_req_pixel_types[k] = TINYEXR_PIXELTYPE_HALF;
    }

    exr_header.pixel_types = tmp_pixel_types.data();
    exr_header.requested_pixel_types = tmp_req_pixel_types.data();

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&exr_image, &exr_header, filename.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return false;
    }
    return true;
  }else if ((mipmap->getDataType() == hikari::MipmapDataType::eF32)) {
    std::vector<F32> pixels[4];
    for (size_t k = 0; k < channel; ++k) {
      pixels[k] = std::vector<F32>(image_size);
      for (size_t j = 0; j < image_size; ++j) {
        pixels[k][j] = ((const F32*)data_ptr)[channel * j + k];
      }
    }
    const F32* image_ptrs[4];
    for (size_t k = 0; k < channel; ++k) {
      image_ptrs[k] = pixels[channel - 1 - k].data();
    }
    exr_image.images = (unsigned char**)image_ptrs;
    exr_image.width  = image->getWidth();
    exr_image.height = image->getHeight();

    std::vector< EXRChannelInfo> channels(channel, EXRChannelInfo{});

    exr_header.num_channels = channel;
    exr_header.channels     = channels.data();

    std::vector<int>         tmp_pixel_types(channel, 0);
    std::vector<int>         tmp_req_pixel_types(channel, 0);
    std::vector<const char*> tmp_channel_names(channel, "");
    for (size_t k = 0; k < channel; ++k) {// R G B A
      channels[channel-1-k].name[0] = channel_names[k][0];
      channels[channel-1-k].name[1] = '\0';
      tmp_pixel_types[k] = TINYEXR_PIXELTYPE_FLOAT;
      tmp_req_pixel_types[k] = TINYEXR_PIXELTYPE_HALF;
    }
    exr_header.pixel_types = tmp_pixel_types.data();
    exr_header.requested_pixel_types = tmp_req_pixel_types.data();
    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&exr_image, &exr_header, filename.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return false;
    }
    return true;
  } else if ((mipmap->getDataType() == hikari::MipmapDataType::eU32)) {
    std::vector<U32> pixels[4];
    for (size_t k = 0; k < channel; ++k) {
      pixels[k] = std::vector<U32>(image_size);
      for (size_t j = 0; j < image_size; ++j) {
        pixels[k][j] = ((const U32*)data_ptr)[channel * j + k];
      }
    }

    const U32* image_ptrs[4];
    for (size_t k = 0; k < channel; ++k) {
      image_ptrs[k] = pixels[channel - 1 - k].data();
    }

    exr_image.images = (unsigned char**)image_ptrs;
    exr_image.width  = image->getWidth();
    exr_image.height = image->getHeight();

    std::vector< EXRChannelInfo> channels(channel, EXRChannelInfo{});

    exr_header.num_channels = channel;
    exr_header.channels = channels.data();

    std::vector<int>         tmp_pixel_types(channel, 0);
    std::vector<int>         tmp_req_pixel_types(channel, 0);
    std::vector<const char*> tmp_channel_names(channel, "");
    for (size_t k = 0; k < channel; ++k) {// R G B A
      channels[channel - 1 - k].name[0] = channel_names[k][0];
      channels[channel - 1 - k].name[1] = '\0';
      tmp_pixel_types[k] = TINYEXR_PIXELTYPE_UINT;
      tmp_req_pixel_types[k] = TINYEXR_PIXELTYPE_UINT;
    }

    exr_header.pixel_types = tmp_pixel_types.data();
    exr_header.requested_pixel_types = tmp_req_pixel_types.data();
    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&exr_image, &exr_header, filename.c_str(), &err);

    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return false;
    }
    return true;
  }



  return false;
}

bool hikari::ImageExporter::saveHdrImage(const String& filename, const MipmapPtr& mipmap)
{
  if (!mipmap) { return false; }
  if (mipmap->getDimension()!= hikari::MipmapDimension::e2D) { return false; }

  auto channel  = mipmap->getChannel();
  auto image    = mipmap->getImage(0);
  auto data_ptr = image->getData();

  if ((mipmap->getDataType() == hikari::MipmapDataType::eF16)) {
    std::vector<hikari::F32> pixels(image->getWidth() * image->getHeight() * channel);
    for (size_t j = 0; j < pixels.size(); ++j) {
      pixels[j] = ((const hikari::F16*)data_ptr)[j];
    }
    stbi_write_hdr(filename.c_str(), image->getWidth(), image->getHeight(), channel, pixels.data());
    return true;
  }
  else if ((mipmap->getDataType() == hikari::MipmapDataType::eF32)) {
    std::vector<hikari::F32> pixels(image->getWidth() * image->getHeight() * channel);
    for (size_t j = 0; j < pixels.size(); ++j) {
      pixels[j] = ((const hikari::F32*)data_ptr)[j];
    }
    stbi_write_hdr(filename.c_str(), image->getWidth(), image->getHeight(), channel, pixels.data());
    return true;
  }

  return false;
}

bool hikari::ImageExporter::savePfmImage(const String& filename, const MipmapPtr& mipmap)
{
  return false;
}
