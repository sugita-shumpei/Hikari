#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>
#if !defined(__CUDACC__)
#include <optional>
#include <variant>
#endif

#define RAY_TYPE_RADIANCE 0
#define RAY_TYPE_OCCLUDED 1
#define RAY_TYPE_COUNT    2

struct CameraData {
  owl::vec3f dir_u;
  float      near_clip;
  owl::vec3f dir_v;
  float      far_clip;
  owl::vec3f dir_w;
  float      dummy0;
  owl::vec3f eye;
  float      dummy1;
};

#define TEXTURE_TYPE_OBJECT  0
#define TEXTURE_TYPE_CHECKER 1

struct TextureData {
  unsigned int          type;
  cudaTextureObject_t object;
  owl::vec3f       colors[2];
  owl::vec3f        to_uv[3];

#if !defined(__CUDACC__)
  void initObject(cudaTextureObject_t object_, const owl::vec3f to_uv_[]) {
    type = TEXTURE_TYPE_OBJECT;
    object = object_;
    to_uv[0] = to_uv_[0];
    to_uv[1] = to_uv_[1];
    to_uv[2] = to_uv_[2];
  }
  void initChecker(const owl::vec3f& color0, const owl::vec3f& color1, const owl::vec3f to_uv_[]) {
    type = TEXTURE_TYPE_CHECKER;
    colors[0] = color0;
    colors[1] = color1;
    to_uv[0] = to_uv_[0];
    to_uv[1] = to_uv_[1];
    to_uv[2] = to_uv_[2];
  }
#else
  __forceinline__ __device__ owl::vec2f transformUV(const owl::vec2f& uv)const {
    auto res = to_uv[0] * uv.x + to_uv[1] * uv.y + to_uv[2];
    return owl::vec2f(res.x / res.z, res.y / res.z);
  }
  __forceinline__ __device__ owl::vec3f sampleObject(float u, float v) const {
    auto uv = transformUV(owl::vec2f(u, v));
    auto col = tex2D<float4>(object, uv.x, uv.y);
    return owl::vec3f(col.x, col.y, col.z);
  }
  __forceinline__ __device__ owl::vec3f sampleChecker(float u, float v) const {
    auto uv = transformUV(owl::vec2f(u, v));
    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);
    auto b0 = (uv.x <= 0.5f);
    auto b1 = (uv.y <= 0.5f);
    return colors[(b0 != b1)];
  }
  __forceinline__ __device__ owl::vec3f sample(float u, float v) const {
    if (type == TEXTURE_TYPE_OBJECT ) { return sampleObject(u, v) ; }
    if (type == TEXTURE_TYPE_CHECKER) { return sampleChecker(u, v); }
    return owl::vec3f(0.0f, 0.0f, 0.0f);
  }
#endif
};

struct SurfaceDiffuseData {
  owl::vec3f reflectance;
};
struct SurfaceConductorData {
  owl::vec3f eta;
  owl::vec3f k;
  owl::vec3f specular_reflectance;
};
struct SurfaceDielectricData {
  float      eta;
  owl::vec3f specular_reflectance;
  owl::vec3f specular_transmittance;
};
struct SurfaceThinDielectricData {
  SurfaceDielectricData dielectric;
};
struct SurfacePlasticData {
  owl::vec3f diffuse_reflectance;
  float      eta;
  owl::vec3f specular_reflectance;
  float      int_fresnel_diffuse_reflectance;
};
struct SurfaceRoughConductorIsotropicData {
  SurfaceConductorData conductor;
  float                alpha;
};
struct SurfaceRoughConductorAnisotropicData {
  SurfaceConductorData conductor;
  float                alpha_u;
  float                alpha_v;
};
struct SurfaceRoughDielectricIsotropicData {
  SurfaceDielectricData dielectric;
  float                 alpha;
};
struct SurfaceRoughDielectricAnisotropicData {
  SurfaceDielectricData dielectric;
  float                 alpha_u;
  float                 alpha_v;
};
struct SurfaceRoughPlasticIsotropicData {
  SurfacePlasticData    plastic;
  float                 alpha;
};

#define SURFACE_TYPE_MASK_DIFFUSE                       0x0100u
#define SURFACE_TYPE_MASK_CONDUCTOR                     0x0200u
#define SURFACE_TYPE_MASK_DIELECTRIC                    0x0400u
#define SURFACE_TYPE_MASK_PLASTIC                       0x0800u
#define SURFACE_TYPE_MASK_ROUGH                         0x1000u
#define SURFACE_TYPE_MASK_ANISOTROPIC                   0x2000u
#define SURFACE_TYPE_MASK_SPECIAL                       0x4000u

#define SURFACE_TYPE_MASK_ALL                           0xFF00u

#define SURFACE_TYPE_DIFFUSE                            SURFACE_TYPE_MASK_DIFFUSE    
#define SURFACE_TYPE_CONDUCTOR                          SURFACE_TYPE_MASK_CONDUCTOR  
#define SURFACE_TYPE_DIELECTRIC                         SURFACE_TYPE_MASK_DIELECTRIC 
#define SURFACE_TYPE_THIN_DIELECTRIC                   (SURFACE_TYPE_MASK_DIELECTRIC |SURFACE_TYPE_MASK_SPECIAL)
#define SURFACE_TYPE_PLASTIC                            SURFACE_TYPE_MASK_PLASTIC
#define SURFACE_TYPE_ROUGH_CONDUCTOR                   (SURFACE_TYPE_MASK_CONDUCTOR  |SURFACE_TYPE_MASK_ROUGH  )
#define SURFACE_TYPE_ROUGH_CONDUCTOR_ISOTROPIC          SURFACE_TYPE_ROUGH_CONDUCTOR
#define SURFACE_TYPE_ROUGH_CONDUCTOR_ANISOTROPIC       (SURFACE_TYPE_ROUGH_CONDUCTOR |SURFACE_TYPE_MASK_ANISOTROPIC)
#define SURFACE_TYPE_ROUGH_DIELECTRIC                  (SURFACE_TYPE_MASK_DIELECTRIC |SURFACE_TYPE_MASK_ROUGH  )
#define SURFACE_TYPE_ROUGH_DIELECTRIC_ISOTROPIC         SURFACE_TYPE_ROUGH_DIELECTRIC
#define SURFACE_TYPE_ROUGH_DIELECTRIC_ANISOTROPIC      (SURFACE_TYPE_ROUGH_DIELECTRIC|SURFACE_TYPE_MASK_ANISOTROPIC)
#define SURFACE_TYPE_ROUGH_PLASTIC                     (SURFACE_TYPE_MASK_PLASTIC    |SURFACE_TYPE_MASK_ROUGH)
#define SURFACE_TYPE_ROUGH_PLASTIC_ISOTROPIC            SURFACE_TYPE_ROUGH_PLASTIC   
#define SURFACE_TYPE_ROUGH_PLASTIC_ANISOTROPIC         (SURFACE_TYPE_ROUGH_PLASTIC   |SURFACE_TYPE_MASK_ANISOTROPIC)

#define SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL               0x0000u
#define SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX               0x0001u

#define SURFACE_TYPE_CONDUCTOR_OPTION_ETA_COL                     0x0000u
#define SURFACE_TYPE_CONDUCTOR_OPTION_ETA_TEX                     0x0001u
#define SURFACE_TYPE_CONDUCTOR_OPTION_K_COL                       0x0000u
#define SURFACE_TYPE_CONDUCTOR_OPTION_K_TEX                       0x0002u
#define SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_COL    0x0000u
#define SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_TEX    0x0004u

#define SURFACE_TYPE_DIELECTRIC_OPTION_ETA_VAL                    0x0000u
#define SURFACE_TYPE_DIELECTRIC_OPTION_ETA_TEX                    0x0001u
#define SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_COL   0x0000u
#define SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_TEX   0x0002u
#define SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_COL 0x0000u
#define SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_TEX 0x0004u

#define SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL       0x0000u
#define SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX       0x0001u
#define SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL      0x0000u
#define SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX      0x0002u

#define SURFACE_TYPE_ROUGH_OPTION_DISTRIBUTION_BECKMAN            0x0000u
#define SURFACE_TYPE_ROUGH_OPTION_DISTRIBUTION_GGX                0x0008u
#define SURFACE_TYPE_ROUGH_OPTION_DISTRIBUTION_PHONG              0x0010u
#define SURFACE_TYPE_ROUGH_OPTION_DISTRIBUTION_MASK               0x0018u

#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_VAL                       0x0000u
#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_TEX                       0x0020u
#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_U_VAL                     0x0000u
#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_U_TEX                     0x0020u
#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_VAL                     0x0000u
#define SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_TEX                     0x0040u

struct SurfaceData {
#if !defined(__CUDACC__)
  void initDiffuse(const std::variant<owl::vec3f,unsigned short>& reflectance)
  {
    type = SURFACE_TYPE_DIFFUSE ;
    if (reflectance.index() == 0) {
      type |= SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL;
      values[0]   = std::get<0>(reflectance).x; values[1] = std::get<0>(reflectance).y; values[2] = std::get<0>(reflectance).z;
    }
    else {
      type |= SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX;
      textures[0] = std::get<1>(reflectance);
    }
  }
  void initConductor(const std::variant<owl::vec3f, unsigned short>& eta, const std::variant<owl::vec3f, unsigned short>& k, const std::variant<owl::vec3f, unsigned short>& specular_reflectance)
  {
    type = SURFACE_TYPE_CONDUCTOR;
    if (eta.index() == 0) {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_ETA_COL;
      values[0] = std::get<0>(eta).x; values[1] = std::get<0>(eta).y; values[2] = std::get<0>(eta).z;
    }
    else {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_ETA_TEX;
      textures[0] = std::get<1>(eta);
    }
    if (k.index() == 0) {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_K_COL;
      values[3] = std::get<0>(k).x; values[4] = std::get<0>(k).y; values[5] = std::get<0>(k).z;
    }
    else {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_K_TEX;
      textures[1] = std::get<1>(k);
    }
    if (specular_reflectance.index() == 0) {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_COL;
      values[6] = std::get<0>(specular_reflectance).x; values[7] = std::get<0>(specular_reflectance).y; values[8] = std::get<0>(specular_reflectance).z;
    }
    else {
      type |= SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_TEX;
      textures[2] = std::get<1>(specular_reflectance);
    }
  }
  void initDielectric(const std::variant<float, unsigned short>& eta, const std::variant<owl::vec3f, unsigned short>& specular_reflectance, const std::variant<owl::vec3f, unsigned short>& specular_transmittance) {
    type = SURFACE_TYPE_DIELECTRIC;
    if (eta.index() == 0) {
      type |= SURFACE_TYPE_DIELECTRIC_OPTION_ETA_VAL;
      values[0] = std::get<0>(eta);
    }
    else {
      type |= SURFACE_TYPE_DIELECTRIC_OPTION_ETA_TEX;
      textures[0] = std::get<1>(eta);
    }
    if (specular_reflectance.index() == 0) {
      type |= SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_COL;
      values[1] = std::get<0>(specular_reflectance).x; values[2] = std::get<0>(specular_reflectance).y; values[3] = std::get<0>(specular_reflectance).z;
    }
    else {
      type |= SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_TEX;
      textures[1] = std::get<1>(specular_reflectance);
    }
    if (specular_transmittance.index() == 0) {
      type     |= SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_COL;
      values[4] = std::get<0>(specular_transmittance).x; values[5] = std::get<0>(specular_transmittance).y; values[6] = std::get<0>(specular_transmittance).z;
    }
    else {
      type       |= SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_TEX;
      textures[2] = std::get<1>(specular_transmittance);
    }
  }
  void initThinDielectric(const std::variant<float, unsigned short>& eta, const std::variant<owl::vec3f, unsigned short>& specular_reflectance, const std::variant<owl::vec3f, unsigned short>& specular_transmittance) {
    initDielectric(eta, specular_reflectance, specular_transmittance);
    type |= SURFACE_TYPE_MASK_SPECIAL;
  }
  void initPlastic(const std::variant<owl::vec3f, unsigned short>& diffuse_reflectance, const std::variant<owl::vec3f, unsigned short>& specular_reflectance, float eta, float int_fresnel_diffuse_reflectance) {
    type = SURFACE_TYPE_PLASTIC;
    if (diffuse_reflectance.index() == 0) {
      type |= SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL;
      values[0] = std::get<0>(diffuse_reflectance).x; values[1] = std::get<0>(diffuse_reflectance).y; values[2] = std::get<0>(diffuse_reflectance).z;
    }
    else {
      type |= SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX;
      textures[0] = std::get<1>(diffuse_reflectance);
    }
    if (specular_reflectance.index() == 0) {
      type |= SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL;
      values[4] = std::get<0>(specular_reflectance).x; values[5] = std::get<0>(specular_reflectance).y; values[6] = std::get<0>(specular_reflectance).z;
    }
    else {
      type |= SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX;
      textures[0] = std::get<1>(specular_reflectance);
    }
    values[3] = eta; values[7] = int_fresnel_diffuse_reflectance;
  }
  void initRoughConductor(
    unsigned int option,
    const std::variant<owl::vec3f, unsigned short>& eta, const std::variant<owl::vec3f, unsigned short>& k, const std::variant<owl::vec3f, unsigned short>& specular_reflectance,
    const std::variant<float, unsigned short>&  alpha_1, const std::optional<std::variant<float, unsigned short>> alpha_2 = std::nullopt) {
    initConductor(eta, k, specular_reflectance);
    if (alpha_1.index()   == 0) {
      values[9] = std::get<0>(alpha_1);
      type |= SURFACE_TYPE_ROUGH_OPTION_ALPHA_VAL;
    } else {
      textures[3] = std::get<1>(alpha_1);
      type   |= SURFACE_TYPE_ROUGH_OPTION_ALPHA_TEX;
    }
    if (!alpha_2) {
      type   |= SURFACE_TYPE_MASK_ROUGH | option;
    }
    else {
      if (alpha_2->index() == 0) {
        values[10] = std::get<0>(*alpha_2);
        type |= SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_VAL;
      } else {
        textures[4] = std::get<1>(*alpha_2);
        type |= SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_TEX;
      }
      type   |= SURFACE_TYPE_MASK_ROUGH | SURFACE_TYPE_MASK_ANISOTROPIC | option;
    }
  }
#else
  __forceinline__ __device__ SurfaceDiffuseData loadDiffuse(const TextureData* texture_buffer, float u, float v) const {
    SurfaceDiffuseData diffuse;
    if ((type & SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL)== SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL){
      diffuse.reflectance.x = values[0];
      diffuse.reflectance.y = values[1];
      diffuse.reflectance.z = values[2];
    }
    if ((type & SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX) == SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX) {
      diffuse.reflectance = texture_buffer[textures[0]].sample(u, v);
    }
    return diffuse;
  }
  __forceinline__ __device__ SurfaceConductorData loadConductor(const TextureData* texture_buffer, float u, float v) const {
    SurfaceConductorData conductor;
    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_ETA_COL) == SURFACE_TYPE_CONDUCTOR_OPTION_ETA_COL) {
      conductor.eta.x = values[0];
      conductor.eta.y = values[1];
      conductor.eta.z = values[2];
    }
    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_ETA_TEX) == SURFACE_TYPE_CONDUCTOR_OPTION_ETA_TEX) {
      conductor.eta = texture_buffer[textures[0]].sample(u, v);
    }

    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_K_COL) == SURFACE_TYPE_CONDUCTOR_OPTION_K_COL) {
      conductor.k.x = values[3];
      conductor.k.y = values[4];
      conductor.k.z = values[5];
    }
    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_K_TEX) == SURFACE_TYPE_CONDUCTOR_OPTION_K_TEX) {
      conductor.k = texture_buffer[textures[1]].sample(u, v);
    }
    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_COL) == SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_COL) {
      conductor.specular_reflectance.x = values[6];
      conductor.specular_reflectance.y = values[7];
      conductor.specular_reflectance.z = values[8];
    }
    if ((type & SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_TEX) == SURFACE_TYPE_CONDUCTOR_OPTION_SPECULAR_REFLECTANCE_TEX) {
      conductor.specular_reflectance = texture_buffer[textures[2]].sample(u, v);
    }
    return conductor;
  }
  __forceinline__ __device__ SurfaceDielectricData loadDielectric(const TextureData* texture_buffer, float u, float v) const {
    SurfaceDielectricData dielectric;
    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_ETA_VAL) == SURFACE_TYPE_DIELECTRIC_OPTION_ETA_VAL) {
      dielectric.eta = values[0];
    }
    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_ETA_TEX) == SURFACE_TYPE_DIELECTRIC_OPTION_ETA_TEX) {
      dielectric.eta = texture_buffer[textures[0]].sample(u, v).x;
    }

    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_COL) == SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_COL) {
      dielectric.specular_reflectance.x = values[1];
      dielectric.specular_reflectance.y = values[2];
      dielectric.specular_reflectance.z = values[3];
    }
    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_TEX) == SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_REFLECTANCE_TEX) {
      dielectric.specular_reflectance = texture_buffer[textures[1]].sample(u, v);
    }
    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_COL) == SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_COL) {
      dielectric.specular_transmittance.x = values[4];
      dielectric.specular_transmittance.y = values[5];
      dielectric.specular_transmittance.z = values[6];
    }
    if ((type & SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_TEX) == SURFACE_TYPE_DIELECTRIC_OPTION_SPECULAR_TRANSMITTANCE_TEX) {
      dielectric.specular_transmittance = texture_buffer[textures[2]].sample(u, v);
    }
    return dielectric;
  }

  __forceinline__ __device__ SurfaceThinDielectricData loadThinDielectric(const TextureData* texture_buffer, float u, float v) const {
    SurfaceThinDielectricData thin;
    thin.dielectric = loadDielectric(texture_buffer, u, v);
    return thin;
  }
  __forceinline__ __device__ SurfacePlasticData loadPlastic(const TextureData* texture_buffer, float u, float v) const {
    SurfacePlasticData plastic;
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL) == SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL) {
      plastic.diffuse_reflectance.x = values[0]; 
      plastic.diffuse_reflectance.y = values[1]; 
      plastic.diffuse_reflectance.z = values[2]; 
    }
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX) == SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX) {
      plastic.diffuse_reflectance  = texture_buffer[textures[0]].sample(u, v);
    }
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL) == SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL) {
      plastic.specular_reflectance.x = values[4];
      plastic.specular_reflectance.y = values[5];
      plastic.specular_reflectance.z = values[6];
    }
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX) == SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX) {
      plastic.specular_reflectance = texture_buffer[textures[1]].sample(u, v);
    }
    plastic.eta = values[3];
    plastic.int_fresnel_diffuse_reflectance = values[7];
    return plastic;
  }
  __forceinline__ __device__ SurfaceRoughConductorIsotropicData loadRoughConductorIsotropic(const TextureData* texture_buffer, float u, float v) const {
    SurfaceRoughConductorIsotropicData rough_isotropic;
    rough_isotropic.conductor = loadConductor(texture_buffer, u, v);
    if ((type & SURFACE_TYPE_ROUGH_OPTION_ALPHA_TEX) == SURFACE_TYPE_ROUGH_OPTION_ALPHA_TEX) {
      rough_isotropic.alpha = texture_buffer[textures[3]].sample(u, v).x;
    }
    else {
      rough_isotropic.alpha = values[9];
    }
    return rough_isotropic;
  }

  __forceinline__ __device__ SurfaceRoughConductorAnisotropicData loadRoughConductorAnisotropic(const TextureData* texture_buffer, float u, float v) const {
    SurfaceRoughConductorAnisotropicData rough_anisotropic;
    rough_anisotropic.conductor = loadConductor(texture_buffer, u, v);
    if ((type & SURFACE_TYPE_ROUGH_OPTION_ALPHA_U_TEX) == SURFACE_TYPE_ROUGH_OPTION_ALPHA_U_TEX) {
      rough_anisotropic.alpha_u = texture_buffer[textures[3]].sample(u, v).x;
    }
    else {
      rough_anisotropic.alpha_u = values[9];
    }
    if ((type & SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_TEX) == SURFACE_TYPE_ROUGH_OPTION_ALPHA_V_TEX) {
      rough_anisotropic.alpha_v = texture_buffer[textures[4]].sample(u, v).x;
    }
    else {
      rough_anisotropic.alpha_v = values[10];
    }
    return rough_anisotropic;
  }
#endif
  unsigned int          type;// float1
  unsigned short textures[6];// float3
  float           values[12];// float12
};

struct LightEnvmapData {
  owl::vec4f       to_local[4];
  cudaTextureObject_t  texture;
  float                  scale;

#if defined(__CUDACC__)
  __forceinline__ __device__ owl::vec3f sample(const owl::vec3f& position) const {
    auto local_pos_x = to_local[0].x * position.x + to_local[0].y * position.y + to_local[0].z * position.z;
    auto local_pos_y = to_local[1].x * position.x + to_local[1].y * position.y + to_local[1].z * position.z;
    auto local_pos_z = to_local[2].x * position.x + to_local[2].y * position.y + to_local[2].z * position.z;

    // +Y = 1 -Y=-1
    auto dir = owl::normalize(owl::vec3f(local_pos_x, local_pos_y, local_pos_z));
    auto u   = atan2f(dir.x,-dir.z) /(2.0f *M_PI);
    if (u < 0.0f) { u += 1.0f; }
    auto v = acosf(dir.y) / M_PI;

    auto color = tex2D<float4>(texture, u, v);
    return owl::vec3f(scale*color.x, scale * color.y, scale * color.z);
  }
#endif
};
struct LightData {
  LightEnvmapData      envmap;
};
struct LaunchParams {
  OptixTraversableHandle tlas;
  TextureData*       textures;
  SurfaceData*       surfaces;
  LightData             light;
};
struct SBTRaygenData {
  CameraData           camera;
  float3*        frame_buffer;
  float3*        accum_buffer;
  int                   width;
  int                  height;
  int                  sample;
};
struct SBTMissData {
};

struct SBTHitgroupData {
  const void*         vertex_buffer;
  const float3*       normal_buffer;
  const float2*       texcrd_buffer;
  const unsigned int*  index_buffer;
  unsigned short           surfaces;
};

struct Onb {
#if defined(__CUDACC__)
  __device__
#endif
    Onb(const owl::vec3f& w_) noexcept : w{ w_ } {
    if (fabsf(w.x) < 0.5f) {
      u = owl::normalize(owl::cross(w, owl::vec3f(1.0f, 0.0f, 0.0f)));
    }
    else if (fabsf(w.y) < 0.5f) {
      u = owl::normalize(owl::cross(w, owl::vec3f(0.0f, 1.0f, 0.0f)));
    }
    else {
      u = owl::normalize(owl::cross(w, owl::vec3f(0.0f, 0.0f, 1.0f)));
    }
    {
      v = owl::normalize(owl::cross(w, u));
    }
  }

#if defined(__CUDACC__)
  __forceinline__ __device__
#endif
    owl::vec3f local_to_world(const owl::vec3f& direction) const {
    return direction.x * u + direction.y * v + direction.z * w;
    }
#if defined(__CUDACC__)
  __forceinline__ __device__
#endif
    owl::vec3f world_to_local(const owl::vec3f& direction) const {
      return owl::vec3f(owl::dot(u,direction),owl::dot(v,direction),owl::dot(w,direction));
    }


  owl::vec3f u;
  owl::vec3f v;
  owl::vec3f w;
};


#if defined(__CUDACC__)
__forceinline__ __device__
owl::vec3f offset_ray(const owl::vec3f& p, const owl::vec3f& n)
{
  constexpr float k_origin      = 1.0f / 32.0f;
  constexpr float k_float_scale = 1.0f / 65536.0f;
  constexpr float k_int_scale   = 256.0f;

  owl::vec3i of_i(k_int_scale * n.x, k_int_scale * n.y, k_int_scale * n.z);
  owl::vec3f p_i(
    __int_as_float(__float_as_int(p.x)+((p.x<0.0f)?-of_i.x:of_i.x)),
    __int_as_float(__float_as_int(p.y)+((p.y<0.0f)?-of_i.y:of_i.y)),
    __int_as_float(__float_as_int(p.z)+((p.z<0.0f)?-of_i.z:of_i.z))
  );
  return owl::vec3f(
    fabsf(p.x) < k_origin ? p.x + k_float_scale * n.x : p_i.x,
    fabsf(p.y) < k_origin ? p.y + k_float_scale * n.y : p_i.y,
    fabsf(p.z) < k_origin ? p.z + k_float_scale * n.z : p_i.z
 );
}

#endif
