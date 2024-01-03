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
