#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <hikari/core/utils.h>
#include <serialized_data.h>
#include <spectrum_data.h>
#include <xml_data.h>
#include <hikari/assets/mitsuba/scene_importer.h>
#include <hikari/camera/orthographic.h>
#include <hikari/camera/perspective.h>
#include <hikari/film/hdr.h>
#include <hikari/film/spec.h>
#include <stb_image_write.h>
#include <tinyxml2.h>
#include <filesystem>
#include <complex>
#include <unordered_set>

static hikari::MitsubaSpectrumData cie1931_x_val = hikari::MitsubaSpectrumData::cie1931_x();
static hikari::MitsubaSpectrumData cie1931_y_val = hikari::MitsubaSpectrumData::cie1931_y();
static hikari::MitsubaSpectrumData cie1931_z_val = hikari::MitsubaSpectrumData::cie1931_z();

std::complex<float> calc_rp(const std::complex<float> &n1, const std::complex<float> &n2, float cos_a)
{
    auto tmp = std::sqrt(n2 * n2 - n1 * n1 * (1 - cos_a * cos_a));
    return (n2 * n2 * cos_a - n1 * tmp) / (n2 * n2 * cos_a + n1 * tmp);
}
std::complex<float> calc_rs(const std::complex<float> &n1, const std::complex<float> &n2, float cos_a)
{
    auto tmp = std::sqrt(n2 * n2 - n1 * n1 * (1 - cos_a * cos_a));
    return (n1 * cos_a - tmp) / (n1 * cos_a + tmp);
}
std::complex<float> calc_tp(const std::complex<float> &n1, const std::complex<float> &n2, float cos_a)
{
    auto tmp = std::sqrt(n2 * n2 - n1 * n1 * (1 - cos_a * cos_a));
    return 2.0f * n1 * n2 * cos_a / (n2 * n2 * cos_a + n1 * tmp);
}
std::complex<float> calc_ts(const std::complex<float> &n1, const std::complex<float> &n2, float cos_a)
{
    auto tmp = std::sqrt(n2 * n2 - n1 * n1 * (1 - cos_a * cos_a));
    return 2.0f * n1 * cos_a / (n1 * cos_a + tmp);
}
// Inputs: Wavelength in nanometers
float        blackbody      (float kelvin, float wavelength) {
  double c1     = (3.74f/3.141592f) * 1.0e+29;//10^-16 ->
  double c2     = 0.0144f;
  auto log_c1 = log(c1);
  auto log_wv = log(wavelength);
  auto log_em = log(expm1(c2 * 1.0e9 / (wavelength * kelvin)));
  return exp(log_c1-5.0*log_wv-log_em);
}
hikari::Vec3 spectrum_to_xyz(const hikari::MitsubaSpectrumData &spec)
{
    auto sum_x     = 0.0f;
    auto sum_y     = 0.0f;
    auto sum_z     = 0.0f;
    auto sum_mat_x = 0.0f;
    auto sum_mat_y = 0.0f;
    auto sum_mat_z = 0.0f;
    for (auto i = 0; i < spec.values.size(); ++i)
    {
        auto wav = spec.wavelengths[i];
        auto val = spec.values[i];

        auto x   = cie1931_x_val.getValue(wav);
        auto y   = cie1931_y_val.getValue(wav);
        auto z   = cie1931_z_val.getValue(wav);

        sum_x     += x * val;
        sum_mat_x += x;
        sum_y     += y * val;
        sum_mat_y += y;
        sum_z     += z * val;
        sum_mat_z += z;

        // std::cout << wav << " " << x << " " << y << " " << z << " " << r << std::endl;
    }
    return hikari::Vec3(sum_x / sum_mat_x, sum_y / sum_mat_y, sum_z / sum_mat_z);
}
hikari::Vec3 blackbody_xyz  (float kelvin) {
  glm::dvec3 v0 = {};
  glm::dvec3 v1 = {};
  for (int i = 360; i < 831; ++i) {
    auto x = cie1931_x_val.getValue(i);
    auto y = cie1931_y_val.getValue(i);
    auto z = cie1931_z_val.getValue(i);
    auto s = blackbody(kelvin, i);
    v0    += glm::dvec3(x * s, y * s, z * s);
    v1    += glm::dvec3(x, y, z);
  }
  return v0 / v1;
}

hikari::Vec3 linear_to_gamma(const hikari::Vec3& v) {
  auto cond = glm::lessThan(v, hikari::Vec3(0.0031308f));
  auto v1   = 12.92f * v;
  auto v2   = (1 + 0.055f) * glm::pow(v, hikari::Vec3(1.0f / 2.4f)) - 0.055f;
  return glm::vec3(cond) * v1 + glm::vec3(cond) * (v2-v1);
}

hikari::Vec3 convert_xyz_to_rgb(const hikari::Vec3& xyz)
{
    auto mat = hikari::Mat3x3(
        hikari::Vec3(2.36461385, -0.51516621, 0.0052037),
        hikari::Vec3(-0.89654057, 1.4264081, -0.01440816),
        hikari::Vec3(-0.46807328, 0.0887581, 1.00920446));
    return mat * xyz;
}
hikari::Vec3 convert_xyz_to_srgb(const hikari::Vec3& xyz) {
  auto mat = hikari::Mat3x3(
    hikari::Vec3(3.240479f ,-0.969256f, 0.055648f),
    hikari::Vec3(-1.537150f, 1.875991f, -0.204043f),
    hikari::Vec3(-0.498535f, 0.041556f, 1.057311f)
  );
  return mat * xyz;
}
hikari::Vec3 schlick           (const hikari::Vec3& f0, float cosine)
{
    auto s = (1 - cosine);
    return f0 + (1.0f - f0) * s * s * s * s * s;
}

int  test() {
  {
    auto filename = R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\tests\serialized\rectangle_normals_uv.serialized)";
    auto serialized_data = hikari::MitsubaSerializedData();
    if (!serialized_data.load(filename, hikari::MitsubaSerializedLoadType::eFull))
    {
      printf("ERR!\n");
      return -1;
    }
    auto& mesh = serialized_data.getMeshes()[0];
  }
  auto write_ref = [](std::string mat_name)
  {
      printf("material %s\n", mat_name.c_str());
      // 絶対屈折率
      auto spectrum_eta = hikari::MitsubaSpectrumData();
      if (!spectrum_eta.load(R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\ior\)" + mat_name + ".eta.spd"))
      {
          printf("ERR!\n");
          return -1;
      }
      // 複素屈折率
      auto spectrum_k = hikari::MitsubaSpectrumData();
      if (!spectrum_k.load(R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\ior\)" + mat_name + ".k.spd"))
      {
          printf("ERR!\n");
          return -1;
      }

      auto get_reflectance = [&spectrum_eta, &spectrum_k](float cosine)
      {
          auto sum_x = 0.0f;
          auto sum_y = 0.0f;
          auto sum_z = 0.0f;
          auto sum_mat_x = 0.0f;
          auto sum_mat_y = 0.0f;
          auto sum_mat_z = 0.0f;

          auto spectrum_ref = spectrum_eta;
          for (auto i = 0; i < spectrum_eta.values.size(); ++i)
          {
              auto wav = spectrum_eta.wavelengths[i];
              auto eta = spectrum_eta.values[i];
              auto k = spectrum_k.values[i];
              auto rp = abs(calc_rp(1.0f, std::complex<float>(eta, k), cosine));
              auto rs = abs(calc_rs(1.0f, std::complex<float>(eta, k), cosine));

              auto r = (rp * rp + rs * rs) * 0.5f;
              spectrum_ref.values[i] = r;
          }
          auto xyz = spectrum_to_xyz(spectrum_ref);
          return xyz;
      };
      {
          auto f0 = convert_xyz_to_rgb(get_reflectance(1.0f));
          std::vector<Byte> pixels(3 * 256 * 256);
          for (hikari::U32 i = 0; i < 256; ++i)
          {
              auto cosine = ((float)i) / 255.0f;
              auto rgb = schlick(f0, cosine);
              for (hikari::U32 j = 0; j < 256; ++j)
              {
                  pixels[3 * (256 * i + j) + 0] = rgb.x * 255;
                  pixels[3 * (256 * i + j) + 1] = rgb.y * 255;
                  pixels[3 * (256 * i + j) + 2] = rgb.z * 255;
              }
              // printf("%f %f %f %f\n", cosine, rgb.r, rgb.g, rgb.b);
          }
          auto filename = "ref_rgb/" + mat_name + ".png";
          stbi_write_png(filename.c_str(), 256, 256, 3, pixels.data(), 3 * 256);
      }
      {
          std::vector<Byte> pixels(3 * 256 * 256);
          for (hikari::U32 i = 0; i < 256; ++i)
          {
              auto cosine = ((float)i) / 255.0f;
              auto xyz = get_reflectance(cosine);
              // auto sum = xyz.x + xyz.y + xyz.z;
              // xyz /= sum;
              auto rgb = convert_xyz_to_rgb(xyz);
              for (hikari::U32 j = 0; j < 256; ++j)
              {
                  pixels[3 * (256 * i + j) + 0] = rgb.x * 255;
                  pixels[3 * (256 * i + j) + 1] = rgb.y * 255;
                  pixels[3 * (256 * i + j) + 2] = rgb.z * 255;
              }
              // printf("%f %f %f %f\n", cosine, rgb.r, rgb.g, rgb.b);
          }
          auto filename = "ref/" + mat_name + ".png";
          stbi_write_png(filename.c_str(), 256, 256, 3, pixels.data(), 3 * 256);
      }
  };
  auto write_tra = [](std::string mat_name)
  {
      printf("material %s\n", mat_name.c_str());
      // 絶対屈折率
      auto spectrum_eta = hikari::MitsubaSpectrumData();
      if (!spectrum_eta.load(R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\ior\)" + mat_name + ".eta.spd"))
      {
          printf("ERR!\n");
          return -1;
      }
      // 複素屈折率
      auto spectrum_k = hikari::MitsubaSpectrumData();
      if (!spectrum_k.load(R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\ior\)" + mat_name + ".k.spd"))
      {
          printf("ERR!\n");
          return -1;
      }

      auto get_transmittance = [&spectrum_eta, &spectrum_k](float cosine)
      {
          auto sum_x = 0.0f;
          auto sum_y = 0.0f;
          auto sum_z = 0.0f;
          auto sum_mat_x = 0.0f;
          auto sum_mat_y = 0.0f;
          auto sum_mat_z = 0.0f;

          auto spectrum_tra = spectrum_eta;
          for (auto i = 0; i < spectrum_eta.values.size(); ++i)
          {
              auto wav = spectrum_eta.wavelengths[i];
              auto eta = spectrum_eta.values[i];
              auto k = spectrum_k.values[i];
              auto rp = abs(calc_tp(1.0f, std::complex<float>(eta, k), cosine));
              auto rs = abs(calc_ts(1.0f, std::complex<float>(eta, k), cosine));

              auto r = (rp * rp + rs * rs) * 0.5f;
              spectrum_tra.values[i] = r;
              // std::cout << wav << " " << x << " " << y << " " << z << " " << r << std::endl;
          }
          auto xyz = spectrum_to_xyz(spectrum_tra);
          return xyz;
      };

      std::vector<Byte> pixels(3 * 256 * 256);
      for (hikari::U32 i = 0; i < 256; ++i)
      {
          auto cosine = ((float)i) / 255.0f;
          auto xyz = get_transmittance(cosine);
          // auto sum = xyz.x + xyz.y + xyz.z;
          // xyz /= sum;
          auto rgb = convert_xyz_to_rgb(xyz);
          for (hikari::U32 j = 0; j < 256; ++j)
          {
              pixels[3 * (256 * i + j) + 0] = rgb.x * 255;
              pixels[3 * (256 * i + j) + 1] = rgb.y * 255;
              pixels[3 * (256 * i + j) + 2] = rgb.z * 255;
          }
          // printf("%f %f %f %f\n", cosine, rgb.r, rgb.g, rgb.b);
      }
      auto filename = "tra/" + mat_name + ".png";
      stbi_write_png(filename.c_str(), 256, 256, 3, pixels.data(), 3 * 256);
  };

  auto path = std::filesystem::path(R"(D:\Users\shumpei\Document\CMake\Hikari\data\data\ior)");
  for (const std::filesystem::directory_entry &de : std::filesystem::directory_iterator(path))
  {
      auto path = de.path();
      auto filename = path.filename().replace_extension().replace_extension();
      write_ref(filename.string());
      write_tra(filename.string());
  }
  {
    std::vector<Byte> pixels(3 * 256 * 256);
    for (size_t i = 0; i < 256; ++i) {
      auto xyz = blackbody_xyz(1500+50*i);
      auto sum = xyz.x + xyz.y + xyz.z;
      xyz /= sum;
      auto rgb = convert_xyz_to_rgb(xyz);
      std::cout << 1500 + i * 50 << "K " << xyz.x << " " << xyz.y << " " << xyz.z << std::endl;
      auto res = fmaxf(rgb.r, fmaxf(rgb.g, rgb.b));
      rgb /= res;
      //sum = rgb.x + rgb.y + rgb.z;
      //rgb /= sum;
      for (hikari::U32 j = 0; j < 256; ++j)
      {
        pixels[3 * (256 * i + j) + 0] = rgb.x * 255;
        pixels[3 * (256 * i + j) + 1] = rgb.y * 255;
        pixels[3 * (256 * i + j) + 2] = rgb.z * 255;
      }
    }
    stbi_write_png("blackbody.png", 256, 256, 3, pixels.data(), 3 * 256);
  }
  {
    std::vector<Byte> pixels(3 * 256 * 256);
    for (size_t i = 0; i < 256; ++i) {
      auto   x = cie1931_x_val.getValue(420.0 + i);
      auto   y = cie1931_y_val.getValue(420.0 + i);
      auto   z = cie1931_z_val.getValue(420.0 + i);
      auto xyz = hikari::Vec3(x, y, z);
      xyz     /= (x + y + z);
      auto rgb = convert_xyz_to_rgb(xyz);
      rgb.x    = fmaxf(0.0f, rgb.x);
      rgb.y    = fmaxf(0.0f, rgb.y);
      rgb.z    = fmaxf(0.0f, rgb.z);
      auto res = fmaxf(rgb.r, fmaxf(rgb.g, rgb.b));
      rgb /= res;
      std::cout << 420.0 + i << " " << rgb.x << " " << rgb.y << " " << rgb.z << std::endl;
      //auto sum = rgb.x + rgb.y + rgb.z;
      //rgb /= sum;
      for (hikari::U32 j = 0; j < 256; ++j)
      {
        pixels[3 * (256 * i + j) + 0] = rgb.x * 255;
        pixels[3 * (256 * i + j) + 1] = rgb.y * 255;
        pixels[3 * (256 * i + j) + 2] = rgb.z * 255;
      }
    }
    stbi_write_png("cie1931.png", 256, 256, 3, pixels.data(), 3 * 256);
  }
}

// 最低限必要なプラグイン
// Integrator: ["path"]
// Sensor    : ["perspective"]
// Emitter   : ["area","envmap","constant"]
// Bsdf      : ["roughplastic","twosided","diffuse","roughdielectric","conductor","bumpmap","thindielectric","dielectric","mask","plastic","roughconductor","conductor","phong","coronamtl","null"]// coronamtlは独自拡張なので無視してよい
// Bsdf.distribution:[GGX, Beckman, Phong]の三つを実装すればよし
// Bsdf.eta,k       : 基本的にはRGBの三値で指定、一パスレンダリングだと扱うのが困難だが...反射のみに限定することで使用可能に
//             拡散: diffuse
//             反射: conductor , roughconductor 
//             屈折: dielectric, roughdielectric, thindielectric
//             複合: plastic   , roughplastic   , phong
//             特殊: mask, bump, twoside, null
// Shape     : ["obj","rectangle","serialized","sphere","cube"]
// Film      : ["hdrfilm"]
// Spectrum  : ["uniform" or "srgb"]
// Texture   : ["bitmap","checkerboard"]
// Medium    : ["homogenous"]
// Rfilter   : ["box","tent","gaussian"]
// Sampler   : ["independent"]

void test2() {
  auto rootpath = std::filesystem::path(R"(D:\Users\shumpei\Document\CMake\Hikari\data\mitsuba)");

  auto integrators = std::unordered_set<std::string>();
  auto shapes    = std::unordered_set<std::string>();
  auto bsdfs     = std::unordered_set<std::string>();
  auto volumes   = std::unordered_set<std::string>();
  auto mediums   = std::unordered_set<std::string>();
  auto sensors   = std::unordered_set<std::string>();
  auto emitters  = std::unordered_set<std::string>();
  auto films     = std::unordered_set<std::string>();
  auto textures  = std::unordered_set<std::string>();
  auto spectrums = std::unordered_set<std::string>();
  auto samplers  = std::unordered_set<std::string>();
  auto rfilters  = std::unordered_set<std::string>();
  auto etas      = std::unordered_set<std::string>();
  auto ks        = std::unordered_set<std::string>();
  auto status    = std::unordered_set<std::string>();

  auto iterateElement = [&integrators, &shapes, &bsdfs, &volumes, &sensors, &emitters, &films, &textures, &spectrums, &samplers, &rfilters, &mediums, &etas, &ks, &status](auto self, tinyxml2::XMLElement* element) ->void {
    for (tinyxml2::XMLNode* node = element->FirstChildElement(); node; node = node->NextSibling()) {
      tinyxml2::XMLElement* nodeElement = node->ToElement();
      if (!nodeElement) { return; }
      if (nodeElement->Value() == std::string_view("float")) {
        auto attribute_name = nodeElement->FindAttribute("name");
        if (attribute_name->Value() == std::string_view("fov")) {
          status.insert("fov");
        }
        if (attribute_name->Value() == std::string_view("principal_point_offset_x")) {
          status.insert("principal_point_offset_x");
        }
        if (attribute_name->Value() == std::string_view("principal_point_offset_y")) {
          status.insert("principal_point_offset_y");
        }
      }
      if (nodeElement->Value() == std::string_view("integrator")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { integrators.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("sensor")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { sensors.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("bsdf")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { bsdfs.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("shape")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { shapes.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("rfilter")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { rfilters.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("film")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { films.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("volume")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { volumes.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("medium")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { mediums.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("sampler")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { samplers.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("emitter")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { emitters.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("spectrum")) {
        std::string type = "";
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) {
          type = attribute_type->Value() + std::string("=");
        }
        auto attribute_value = nodeElement->FindAttribute("value");
        if (attribute_value) { spectrums.insert(type + attribute_value->Value()); }
        auto attribute_name = nodeElement->FindAttribute("name");
        if (attribute_name) {
          if (attribute_name->Value() == std::string_view("eta")) {
            auto attribute_value = nodeElement->FindAttribute("value");
            if (attribute_value) { etas.insert(attribute_value->Value()); }
          }
          if (attribute_name->Value() == std::string_view("k")) {
            auto attribute_value = nodeElement->FindAttribute("value");
            if (attribute_value) { ks.insert(attribute_value->Value()); }
          }
        }
      }
      if (nodeElement->Value() == std::string_view("rgb")) {
        auto attribute_name = nodeElement->FindAttribute("name");
        if (attribute_name) {
          if (attribute_name->Value() == std::string_view("eta")) {
            auto attribute_value = nodeElement->FindAttribute("value");
            if (attribute_value) { etas.insert(attribute_value->Value()); }
          }
          if (attribute_name->Value() == std::string_view("k")) {
            auto attribute_value = nodeElement->FindAttribute("value");
            if (attribute_value) { ks.insert(attribute_value->Value()); }
          }
        }
      }
      if (nodeElement->Value() == std::string_view("texture")) {
        auto attribute_type = nodeElement->FindAttribute("type");
        if (attribute_type) { textures.insert(attribute_type->Value()); }
      }
      if (nodeElement->Value() == std::string_view("string")) {
        auto attribute_name = nodeElement->FindAttribute("name");
        if (attribute_name->Value() == std::string_view("focal_length")) {
          status.insert("focal_length");
        }
        if (attribute_name->Value() == std::string_view("fov_axis")) {
          status.insert("fov_axis");
          auto attribute_value = nodeElement->FindAttribute("value");
          if (attribute_value) {
            status.insert("fov_axis=" + std::string(attribute_value->Value()));
          }
        }
      }
      self(self, nodeElement);
    }
    };

 for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(rootpath)) {

      if (!entry.exists() || !entry.is_directory()) { continue; }
      auto filepath = entry.path() / "scene.xml";
      auto filepath_string = filepath.string();
      if (!std::filesystem::exists(filepath)) { continue; }
      auto document = tinyxml2::XMLDocument();
      if (document.LoadFile(filepath_string.c_str()) != tinyxml2::XML_SUCCESS) { continue; }
      auto docHandle = tinyxml2::XMLHandle(&document);
      auto element = docHandle.FirstChildElement("scene").ToElement();
      iterateElement(iterateElement, element);
    }
}


int  main()
{
  auto filepath = std::filesystem::path(R"(D:\Users\shumpei\Document\CMake\Hikari\data\mitsuba\pool\scene.xml)");
  auto importer = hikari::MitsubaSceneImporter::create();
  auto scene    = importer->loadScene(filepath.string());
  auto cameras  = scene->getCameras();
  auto lights   = scene->getLights();
  auto shapes   = scene->getShapes();
  return 0;
}
