#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>

struct CameraData {
  owl::vec3f dir_u;
  float  near_clip;
  owl::vec3f dir_v;
  float  far_clip;
  owl::vec3f dir_w;
  float  dummy0;
  owl::vec3f eye;
  float  dummy1;
};

struct LightEnvmapData {
  owl::vec4f       to_local[4];
  cudaTextureObject_t  texture;
  float                  scale;

#if defined(__CUDACC__)
  __forceinline__ __device__ owl::vec3f sample(const owl::vec3f& position) const {
    auto local_pos_x = to_local[0].x * position.x + to_local[0].y * position.y + to_local[0].z * position.z + to_local[0].w;
    auto local_pos_y = to_local[1].x * position.x + to_local[1].y * position.y + to_local[1].z * position.z + to_local[1].w;
    auto local_pos_z = to_local[2].x * position.x + to_local[2].y * position.y + to_local[2].z * position.z + to_local[2].w;
    auto local_pos_w = to_local[3].x * position.x + to_local[3].y * position.y + to_local[3].z * position.z + to_local[3].w;

    local_pos_x /= local_pos_w;
    local_pos_y /= local_pos_w;
    local_pos_z /= local_pos_w;

    // +Y = 1 -Y=-1
    auto dir = owl::normalize(owl::vec3f(local_pos_x, local_pos_y, local_pos_z));
    auto u   = atan2f(dir.z, dir.x) /(2.0f *M_PI) + 0.25f;
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
};
