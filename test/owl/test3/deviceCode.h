#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

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
  owl::vec3f           reflectance;
};
struct SurfacePlasticData {
  owl::vec3f   diffuse_reflectance;
  float                        eta;
  owl::vec3f  specular_reflectance;
  float int_fresnel_diffuse_reflectance;
};

#define SURFACE_TYPE_DIFFUSE 0x1000u
#define SURFACE_TYPE_PLASTIC 0x2000u
#define SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL 0x0000u
#define SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX 0x0001u
#define SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL 0x0000u
#define SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX 0x0001u
#define SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL 0x0000u
#define SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX 0x0002u
struct SurfaceData {
#if !defined(__CUDACC__)
  void initDiffuse(const owl::vec3f& reflectance_col)
  {
    type = SURFACE_TYPE_DIFFUSE | SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_COL;
    values[0] = reflectance_col.x; values[1] = reflectance_col.y; values[2] = reflectance_col.z;
  }
  void initDiffuse(unsigned short    reflectance_tex)
  {
    type = SURFACE_TYPE_DIFFUSE | SURFACE_TYPE_DIFFUSE_OPTION_REFLECTANCE_TEX;
    textures[0] = reflectance_tex;
  }
  void initPlastic(const owl::vec3f& diffuse_reflectance_col, const owl::vec3f& specular_reflectance_col, float eta, float int_fresnel_diffuse_reflectance)
  {
    type = SURFACE_TYPE_PLASTIC | SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL | SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL;
    values[0] = diffuse_reflectance_col.x ; values[1] = diffuse_reflectance_col.y ; values[2] = diffuse_reflectance_col.z;
    values[4] = specular_reflectance_col.x; values[5] = specular_reflectance_col.y; values[6] = specular_reflectance_col.z;
    values[3] = eta; values[7] = int_fresnel_diffuse_reflectance;
  }
  void initPlastic(unsigned short    diffuse_reflectance_tex, const owl::vec3f& specular_reflectance_col, float eta, float int_fresnel_diffuse_reflectance) {
    type = SURFACE_TYPE_PLASTIC | SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX | SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_COL;
    textures[0] =  diffuse_reflectance_tex;
    values[4] = specular_reflectance_col.x; values[5] = specular_reflectance_col.y; values[6] = specular_reflectance_col.z;
    values[3] = eta; values[7] = int_fresnel_diffuse_reflectance;
  }
  void initPlastic(const owl::vec3f& diffuse_reflectance_col, unsigned short    specular_reflectance_tex, float eta, float int_fresnel_diffuse_reflectance) {
    type = SURFACE_TYPE_PLASTIC | SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL | SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX;
    textures[1] = specular_reflectance_tex;
    values[0] = diffuse_reflectance_col.x; values[1] = diffuse_reflectance_col.y; values[2] = diffuse_reflectance_col.z;
    values[3] = eta; values[7] = int_fresnel_diffuse_reflectance;
  }
  void initPlastic(unsigned short    diffuse_reflectance_tex, unsigned short    specular_reflectance_tex, float eta, float int_fresnel_diffuse_reflectance) {
    type = SURFACE_TYPE_PLASTIC | SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX | SURFACE_TYPE_PLASTIC_OPTION_SPECULAR_REFLECTANCE_TEX;
    textures[0] = diffuse_reflectance_tex; textures[1] = specular_reflectance_tex;
    values[3] = eta; values[7] = int_fresnel_diffuse_reflectance;
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
  __forceinline__ __device__ SurfacePlasticData loadPlastic(const TextureData* texture_buffer, float u, float v) const {
    SurfacePlasticData plastic;
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL) == SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_COL) {
      plastic.diffuse_reflectance.x = values[0]; 
      plastic.diffuse_reflectance.y = values[1]; 
      plastic.diffuse_reflectance.z = values[2]; 
    }
    if ((type & SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX) == SURFACE_TYPE_PLASTIC_OPTION_DIFFUSE_REFLECTANCE_TEX) {
      plastic.diffuse_reflectance  = texture_buffer[textures[0]].sample(u, v);
      plastic.specular_reflectance = texture_buffer[textures[1]].sample(u, v);
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
#endif
  unsigned int          type;
  unsigned short textures[4];
  float            values[9];
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
    owl::vec3f local(const owl::vec3f& direction) const {
    return direction.x * u + direction.y * v + direction.z * w;
  }


  owl::vec3f u;
  owl::vec3f v;
  owl::vec3f w;
};

