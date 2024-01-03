#include "deviceCode.h"

extern "C" { __constant__ LaunchParams optixLaunchParams;  }

struct Payload {
  float3 color;
};

extern "C" {
  __global__ void __raygen__default() {
    auto sbt_rg_data= reinterpret_cast<const SBTRaygenData*>(optixGetSbtDataPointer());
    auto launch_idx = optixGetLaunchIndex();

    owl::LCG<24> random = {};
    random.init(launch_idx.x * launch_idx.y + launch_idx.x, sbt_rg_data->sample);

    owl::vec3f res = owl::vec3f(0.0f, 0.0f, 0.0f);

    auto frame_index = launch_idx.y * sbt_rg_data->width + launch_idx.x;
    for (int i = 0; i < 100; ++i) {
      auto uv = make_float2(
        ((float)launch_idx.x+ (float)random()-0.5f) / (float)sbt_rg_data->width,
        ((float)launch_idx.y+ (float)random()-0.5f) / (float)sbt_rg_data->height
      );

      uv.x = 2.0f * uv.x - 1.0f;
      uv.y = 2.0f * uv.y - 1.0f;

      auto pos1 = owl::vec3f(sbt_rg_data->camera.transform(make_float3(uv.x, uv.y, -1.0f)));
      auto pos2 = owl::vec3f(sbt_rg_data->camera.transform(make_float3(uv.x, uv.y, +1.0f)));

      auto dir = owl::normalize(pos2 - pos1);
      auto len = owl::length(pos2    - pos1);
      owl::RayT<0, 1> ray(pos1, dir, 0.0f, len);

      Payload payload;
      payload.color = make_float3(0.0f, 0.0f, 0.0f);
      owl::traceRay(optixLaunchParams.tlas, ray, payload, OPTIX_RAY_FLAG_NONE);

      res += owl::vec3f(payload.color);
    }
    res *= static_cast<float>(1.0f / 100.0f);
    auto old_color = owl::vec3f(sbt_rg_data->accum_buffer[frame_index]);
    old_color     += res;
    sbt_rg_data->frame_buffer[frame_index] = old_color/static_cast<float>(sbt_rg_data->sample+1.0f);
    sbt_rg_data->accum_buffer[frame_index] = old_color;
  }
  __global__ void __miss__default()   {
    auto& payload = owl::getPRD<Payload>();
    payload.color = make_float3(1.0f,0.0f,0.0f);
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
    auto origin        = optixGetWorldRayOrigin();
    auto tmin          = optixGetRayTmin();

    auto normal0 = owl::vec3f(normal_buffer[tri_index.x]);
    auto normal1 = owl::vec3f(normal_buffer[tri_index.y]);
    auto normal2 = owl::vec3f(normal_buffer[tri_index.z]);

    auto texcrd0 = owl::vec2f(texcrd_buffer[tri_index.x]);
    auto texcrd1 = owl::vec2f(texcrd_buffer[tri_index.y]);
    auto texcrd2 = owl::vec2f(texcrd_buffer[tri_index.z]);

    auto normal = (1.0f - barycentrics.x - barycentrics.y) * normal0 + barycentrics.x * normal1 + barycentrics.y * normal2;
    auto texcrd = (1.0f - barycentrics.x - barycentrics.y) * texcrd0 + barycentrics.x * texcrd1 + barycentrics.y * texcrd2;

    auto  color   = 0.5f * normal + owl::vec3f(0.5f, 0.5f, 0.5f);
    auto& payload = owl::getPRD<Payload>();
    payload.color = color;
  }
}
