#include "deviceCode.h"

extern "C" { __constant__ LaunchParams optixLaunchParams;  }

struct Payload {
  float3 color;
};

extern "C" {
  __global__ void __raygen__default() {
    auto sbt_rg_data= reinterpret_cast<const SBTRaygenData*>(optixGetSbtDataPointer());
    auto launch_idx = optixGetLaunchIndex();

    auto uv = make_float2(
      (float)launch_idx.x / (float)sbt_rg_data->width ,
      (float)launch_idx.y / (float)sbt_rg_data->height
    );

    uv.x = 2.0f * uv.x - 1.0f;
    uv.y = 2.0f * uv.y - 1.0f;

    auto org   = sbt_rg_data->camera.eye;
    auto dir   =(sbt_rg_data->camera.dir_w + uv.x * sbt_rg_data->camera.dir_u+ uv.y * sbt_rg_data->camera.dir_v);

    dir        = owl::normalize(dir);
    auto tmin  = sbt_rg_data->camera.near_clip;
    auto tmax  = sbt_rg_data->camera.far_clip;

    owl::RayT<0,1> ray(org, dir, tmin, tmax);

    Payload payload;
    payload.color = make_float3(0.0f, 0.0f, 0.0f);
    owl::traceRay(optixLaunchParams.tlas, ray, payload, OPTIX_RAY_FLAG_NONE);

    auto frame_index = launch_idx.y * sbt_rg_data->width + launch_idx.x;

    sbt_rg_data->frame_buffer[frame_index] = payload.color;
    sbt_rg_data->accum_buffer[frame_index] = payload.color;
  }
  __global__ void __miss__default()   {
    auto sbt_ms_data = reinterpret_cast<const SBTMissData*>(optixGetSbtDataPointer());
    auto& payload = owl::getPRD<Payload>();
    auto pos = owl::vec3f(optixGetWorldRayOrigin())+ optixGetRayTmax() * owl::vec3f(optixGetWorldRayDirection());
    payload.color = optixLaunchParams.light.envmap.sample(pos);
  }
  __global__ void __closesthit__default_triangle() {
    auto sbt_hg_data   = reinterpret_cast<const SBTHitgroupData*>(optixGetSbtDataPointer());
    auto vertex_buffer = reinterpret_cast<const float3*>(sbt_hg_data->vertex_buffer);
    auto normal_buffer = sbt_hg_data->normal_buffer;
    auto texcrd_buffer = sbt_hg_data->texcrd_buffer;
    auto index_buffer  = reinterpret_cast<const uint3*>(sbt_hg_data->index_buffer);
    auto barycentrics  = optixGetTriangleBarycentrics();
    auto prim_index    = optixGetPrimitiveIndex();
    auto tri_index     = index_buffer[prim_index];
    auto origin        = owl::vec3f(optixGetWorldRayOrigin());
    auto direction     = owl::vec3f(optixGetWorldRayDirection());
    auto tmin          = optixGetRayTmin();
    auto position      = origin + tmin*direction;
    auto normal0       = owl::vec3f(normal_buffer[tri_index.x]);
    auto normal1       = owl::vec3f(normal_buffer[tri_index.y]);
    auto normal2       = owl::vec3f(normal_buffer[tri_index.z]);
    auto texcrd0       = owl::vec2f(texcrd_buffer[tri_index.x]);
    auto texcrd1       = owl::vec2f(texcrd_buffer[tri_index.y]);
    auto texcrd2       = owl::vec2f(texcrd_buffer[tri_index.z]);

    auto normal   = owl::vec3f((1.0f - barycentrics.x - barycentrics.y) * normal0 + barycentrics.x * normal1 + barycentrics.y * normal2);
    auto texcrd   = (1.0f - barycentrics.x - barycentrics.y) * texcrd0 + barycentrics.x * texcrd1 + barycentrics.y * texcrd2;

    auto reflection  = direction - 2.0f * owl::dot(direction, normal) * normal;
    auto reflect_pos = position + 1000.0f * reflection;


    auto  color   = optixLaunchParams.light.envmap.sample(reflect_pos);
    auto& payload = owl::getPRD<Payload>();
    payload.color = color;
  }
}
