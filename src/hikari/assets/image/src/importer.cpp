#define STB_IMAGE_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION 
#include <hikari/assets/image/importer.h>
#include <hikari/core/utils.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <stb_image.h>
#include <tinyexr.h>

auto hikari::ImageImporter::getInstance() -> ImageImporter&
{
  static ImageImporter importer;
  return importer;
}

hikari::ImageImporter::~ImageImporter()
{
}

hikari::MipmapPtr hikari::ImageImporter::load(const String& filename)
{
  auto iter = m_mipmaps.find(filename);
  if (iter != std::end(m_mipmaps)) { return iter->second; }
  auto path = std::filesystem::path(filename);
  auto ext = path.extension().string();
  if ((ext == ".jpg" ) ||
      (ext == ".jpeg") ||
      (ext == ".png" ) ||
      (ext == ".tga" ) ||
      (ext == ".gif" ) ||
      (ext == ".bmp" ) ||
      (ext == ".psd" ) ||
      (ext == ".ppm" ) ||
      (ext == ".pnm" ) ||
      (ext == ".pgm" )) { return loadStbImage(filename); }
  if ((ext == ".hdr" )) { return loadHdrImage(filename); }
  if ((ext == ".exr" )) { return loadExrImage(filename); }
  if ((ext == ".pfm" )) { return loadPfmImage(filename); }
  return nullptr;
}

void hikari::ImageImporter::free(const String& filename)
{
  if (m_mipmaps.erase(filename) > 0) { return; }
  std::string filepath = "";
  for (auto& [path_str, mipmap] : m_mipmaps) {
    auto path = std::filesystem::path(path_str);
    if (path.filename() != filename) { continue; }
    filepath = path.string();
    break;
  }
  if (filepath.empty()) { return; }
  m_mipmaps.erase(filepath);
}

void hikari::ImageImporter::clear()
{
  m_mipmaps.clear();
}

auto hikari::ImageImporter::get(const String& filename) const -> MipmapPtr
{
  auto iter = m_mipmaps.find(filename);
  if (iter != std::end(m_mipmaps)) { return iter->second; }
  std::string filepath = "";
  for (auto& [path_str, mipmap] : m_mipmaps) {
    auto path = std::filesystem::path(path_str);
    if (path.filename() != filename) { continue; }
    filepath = path.string();
    break;
  }
  if (filepath.empty()) { return nullptr; }
  iter = m_mipmaps.find(filepath);
  if (iter != std::end(m_mipmaps)) { return iter->second; }
  return nullptr;
}

auto hikari::ImageImporter::loadStbImage(const String& filename) -> MipmapPtr
{
  int  width, height, comp;
  auto pixels = stbi_load(filename.c_str(), &width, &height, &comp, 0);
  if (!pixels) { return nullptr; }

  hikari::MipmapImageDesc desc;
  desc.p_data         = (const Byte*)pixels;
  desc.width_in_bytes = width * comp;
  desc.height         = height;
  desc.depth_or_layers= 1;
  desc.x = 0;
  desc.y = 0;
  desc.z = 0;

  auto res = Mipmap::create2D(hikari::MipmapDataType::eU8, comp, 1, width, height, { desc },true);
  m_mipmaps.insert({ filename,res });
  stbi_image_free(pixels);
  return res;
}

auto hikari::ImageImporter::loadHdrImage(const String& filename) -> MipmapPtr
{
  int  width, height, comp;
  auto pixels = stbi_loadf(filename.c_str(), &width, &height, &comp, 0);
  if (!pixels) { return nullptr; }

  hikari::MipmapImageDesc desc;
  desc.p_data          = (const Byte*)pixels;
  desc.width_in_bytes  = width * comp * sizeof(F32);
  desc.height          = height;
  desc.depth_or_layers = 1;
  desc.x = 0;
  desc.y = 0;
  desc.z = 0;

  auto res = Mipmap::create2D(hikari::MipmapDataType::eF32, comp, 1, width, height, { desc }, true);
  m_mipmaps.insert({ filename,res });
  stbi_image_free(pixels);
  return res;
}

auto hikari::ImageImporter::loadExrImage(const String& filename) -> MipmapPtr
{
  const char* filename_cstr = filename.c_str();
  EXRVersion exr_version;
  int ret = ParseEXRVersionFromFile(&exr_version, filename_cstr);
  if (ret != 0) {
    fprintf(stderr, "Invalid EXR file: %s\n", filename_cstr);
    return nullptr;
  }
  if (!exr_version.multipart) {
    EXRHeader exr_header;
    InitEXRHeader(&exr_header);
    const char* err = nullptr; // or `nullptr` in C++11 or later.
    ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, filename_cstr, &err);
    if (ret != 0) {
      fprintf(stderr, "Parse EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return nullptr;
    }

    EXRImage exr_image; 
    InitEXRImage(&exr_image);
    ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename_cstr, &err);
    if (ret != 0) {
      fprintf(stderr, "Load EXR err: %s\n", err);
      FreeEXRHeader(&exr_header);
      FreeEXRErrorMessage(err); // free's buffer for an error message
      return nullptr;
    }

    if (exr_header.num_channels == 1) { return nullptr; }

    auto channel_indices = std::vector<size_t>();
    channel_indices.reserve(exr_header.num_channels);
    for (size_t c = 0; c < exr_header.num_channels; ++c) {
      auto channel_name = std::string_view(exr_header.channels[c].name);
      if (channel_name == "R") { channel_indices.push_back(0); }
      else if (channel_name == "G") { channel_indices.push_back(1); }
      else if (channel_name == "B") { channel_indices.push_back(2); }
      else if (channel_name == "A") { channel_indices.push_back(3); }
      else { throw std::runtime_error("Failed To Support Channel Name!"); }
    }

    auto pixel_type = exr_header.requested_pixel_types[0];
    for (size_t c = 0; c < exr_header.num_channels; ++c) { if (pixel_type!=exr_header.requested_pixel_types[c]) { return nullptr; } }

    auto data_type = hikari::MipmapDataType();
    if (pixel_type == TINYEXR_PIXELTYPE_HALF ) { data_type = hikari::MipmapDataType::eF16; }
    if (pixel_type == TINYEXR_PIXELTYPE_FLOAT) { data_type = hikari::MipmapDataType::eF32; }
    if (pixel_type == TINYEXR_PIXELTYPE_UINT ) { data_type = hikari::MipmapDataType::eU32; }
    if (data_type  == hikari::MipmapDataType()) { return nullptr; }
    if (data_type  == hikari::MipmapDataType::eF16) {
      std::vector<F16> pixels(exr_image.width * exr_image.height * exr_image.num_channels, 0.0f);
      for (size_t c = 0; c < exr_header.num_channels; ++c) {
        auto channel_idx = channel_indices[c];
        for (size_t j = 0; j < exr_image.height; ++j) {
          for (size_t i = 0; i < exr_image.width; ++i) {
            pixels[exr_header.num_channels * (exr_image.width * j + i) + channel_idx] = ((const F16*)exr_image.images[c])[(exr_image.width * j + i)];
          }
        }
      }

      hikari::MipmapImageDesc desc = {};
      desc.p_data = pixels.data();
      desc.width_in_bytes = exr_image.width * sizeof(F16) * exr_header.num_channels;
      desc.height = exr_image.height;
      desc.depth_or_layers = 1;
      desc.x = 0; desc.y = 0; desc.z = 0;

      auto res = hikari::Mipmap::create2D(data_type, exr_header.num_channels, 1, exr_image.width, exr_image.height, { desc }, true);
      return res;
    }
    if (data_type  == hikari::MipmapDataType::eF32) {
      std::vector<F32> pixels(exr_image.width * exr_image.height * exr_image.num_channels,0.0f);
      for (size_t c = 0; c < exr_header.num_channels; ++c) {
        auto channel_idx = channel_indices[c];
        for (size_t j = 0; j < exr_image.height; ++j) {
          for (size_t i = 0; i < exr_image.width; ++i) {
            pixels[exr_header.num_channels * (exr_image.width * j + i) + channel_idx] = ((const F32*)exr_image.images[c])[(exr_image.width * j + i)];
          }
        }
      }

      hikari::MipmapImageDesc desc = {};
      desc.p_data          = pixels.data();
      desc.width_in_bytes  = exr_image.width * sizeof(F32) * exr_header.num_channels;
      desc.height          = exr_image.height;
      desc.depth_or_layers = 1;
      desc.x = 0; desc.y = 0; desc.z = 0;

      auto res = hikari::Mipmap::create2D(data_type, exr_header.num_channels, 1, exr_image.width, exr_image.height, { desc }, true);
      m_mipmaps.insert({ filename,res });
      return res;
    }
    if (data_type  == hikari::MipmapDataType::eU32) {
      std::vector<U32> pixels(exr_image.width * exr_image.height * exr_image.num_channels, 0.0f);
      for (size_t c = 0; c < exr_header.num_channels; ++c) {
        auto channel_idx = channel_indices[c];
        for (size_t j = 0; j < exr_image.height; ++j) {
          for (size_t i = 0; i < exr_image.width; ++i) {
            pixels[exr_header.num_channels * (exr_image.width * j + i) + channel_idx] = ((const U32*)exr_image.images[c])[(exr_image.width * j + i)];
          }
        }
      }

      hikari::MipmapImageDesc desc = {};
      desc.p_data = pixels.data();
      desc.width_in_bytes = exr_image.width * sizeof(U32) * exr_header.num_channels;
      desc.height = exr_image.height;
      desc.depth_or_layers = 1;
      desc.x = 0; desc.y = 0; desc.z = 0;

      auto res = hikari::Mipmap::create2D(data_type, exr_header.num_channels, 1, exr_image.width, exr_image.height, { desc }, true);
      m_mipmaps.insert({ filename,res });
      return res;
    }
    return nullptr;
  }
  else {
    return nullptr;
  }
}

auto hikari::ImageImporter::loadPfmImage(const String& filename) -> MipmapPtr
{
  auto pfm_file = std::ifstream(filename, std::ios::binary);
  if (pfm_file.fail()) { return nullptr; }
  pfm_file.seekg(0L, std::ios::end);
  auto pfm_file_size = (size_t)pfm_file.tellg();
  pfm_file.seekg(0L, std::ios::beg);


  int data_count = 0;
  {
    std::string sentence;
    std::getline(pfm_file, sentence);
    if (sentence[0] == 'P' && sentence[1] == 'F') {
      data_count = 3;
    }
    else if (sentence[0] == 'P' && sentence[1] == 'f') {
      data_count = 1;
    }
    else {
      return nullptr;
    }
  }

  int width = 0; int height = 0;
  {
    std::string sentence;
    std::getline(pfm_file, sentence);
    auto strs = hikari::splitString(sentence, ' ');
    try {
      width = std::stoi(strs[0]);
      height = std::stoi(strs[1]);
    }
    catch (std::invalid_argument&) { return nullptr; }
    catch (std::out_of_range&) { return nullptr; }
  }
  bool is_little_endian = false;
  {
    std::string sentence;
    std::getline(pfm_file, sentence);
    float value = 0.0f;
    try {
      value = std::stof(sentence);
    }
    catch (std::invalid_argument&) { return nullptr; }
    catch (std::out_of_range&) { return nullptr; }
    if (value > 0.0f) { is_little_endian = false; }
    else { is_little_endian = true; }
  }
  // BIG ENDIANの場合, 変換が必要
  std::vector<float>  flt_data(width * height * data_count);
  if (!is_little_endian) {
    std::vector<char> big_endian_data;
    std::vector<char> lit_endian_data;
    big_endian_data.resize(width * height * sizeof(float) * data_count);
    lit_endian_data.resize(width * height * sizeof(float) * data_count);
    pfm_file.read(big_endian_data.data(), big_endian_data.size());
    for (size_t i = 0; i < big_endian_data.size() / 4; ++i) {
      lit_endian_data[4 * i + 0] = big_endian_data[4 * i + 3];// 4 BYTE ENDIAN変換
      lit_endian_data[4 * i + 1] = big_endian_data[4 * i + 2];
      lit_endian_data[4 * i + 2] = big_endian_data[4 * i + 1];
      lit_endian_data[4 * i + 3] = big_endian_data[4 * i + 0];
    }
    std::vector<float>  flt_data2(width * height * data_count);
    std::memcpy(flt_data2.data(), lit_endian_data.data(), lit_endian_data.size());
    for (size_t i = 0; i < height; ++i) {
      std::memcpy(flt_data.data() + i * width * data_count, flt_data2.data() + (height - 1 - i) * width * data_count, width * data_count * sizeof(float));
    }
  }
  else {
    std::vector<char> lit_endian_data;
    lit_endian_data.resize(width * height * sizeof(float) * data_count);
    pfm_file.read(lit_endian_data.data(), lit_endian_data.size());
    std::vector<float>  flt_data2(width * height * data_count);
    std::memcpy(flt_data2.data(), lit_endian_data.data(), lit_endian_data.size());
    for (size_t i = 0; i < height; ++i) {
      std::memcpy(flt_data.data() + i * width * data_count, flt_data2.data() + (height - 1 - i) * width * data_count, width * data_count * sizeof(float));
    }
  }

  hikari::MipmapImageDesc desc;
  desc.p_data          = (const Byte*)flt_data.data();
  desc.width_in_bytes  = width * data_count * sizeof(F32);
  desc.height          = height;
  desc.depth_or_layers = 1;
  desc.x = 0;
  desc.y = 0;
  desc.z = 0;

  auto res = Mipmap::create2D(hikari::MipmapDataType::eF32, data_count, 1, width, height, { desc }, true);
  m_mipmaps.insert({ filename,res });
  return res;
}
