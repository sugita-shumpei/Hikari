#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

struct CameraData {
  //owl::vec3f dir_u;
  //float  near_clip;
  //owl::vec3f dir_v;
  //float   far_clip;
  //owl::vec3f dir_w;
  //float     dummy0;
  //owl::vec3f   eye;
  //float     dummy1;
  float4   matrix[4];
#if defined(__CUDACC__)
  __device__ auto transform(const float3& screen) const -> float3 {
    float4 res = {};
    res.x = matrix[0].x * screen.x + matrix[0].y * screen.y + screen.z * matrix[0].z + matrix[0].w;
    res.y = matrix[1].x * screen.x + matrix[1].y * screen.y + screen.z * matrix[1].z + matrix[1].w;
    res.z = matrix[2].x * screen.x + matrix[2].y * screen.y + screen.z * matrix[2].z + matrix[2].w;
    res.w = matrix[3].x * screen.x + matrix[3].y * screen.y + screen.z * matrix[3].z + matrix[3].w;
    return make_float3(res.x / res.w, res.y / res.w, res.z / res.w);
  }
#endif
};

struct LaunchParams {
  OptixTraversableHandle tlas;
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
