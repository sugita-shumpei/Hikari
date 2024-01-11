#include "deviceCode.h"

extern "C" { __constant__ LaunchParams optixLaunchParams;  }

struct Payload {
  owl::LCG<24> random;
  float3       color;
};

__forceinline__ __device__ float fresnel(float eta, float cos_in_2, float cos_out_2) {
  if (cos_out_2 < 0.0f) { return 1.0f; }
  float cos_in = sqrtf(cos_in_2);
  float cos_out= sqrtf(cos_out_2);
  float rp = (eta * cos_in - cos_out) / (eta * cos_in + cos_out);
  float rs = (eta * cos_out - cos_in) / (eta * cos_out + cos_in);
  return 0.5f * (rp * rp + rs * rs);
}

__forceinline__ __device__ owl::vec3f random_in_pdf_cosine(owl::LCG<24>& random) {
  float cos_tht = sqrtf(1 - random());
  float sin_tht = sqrtf(fmaxf(1 - cos_tht * cos_tht,0.0f));
  float phi = 2.0f * M_PI * random();
  float cos_phi = cosf(phi);
  float sin_phi = sinf(phi);
  return { sin_tht * cos_phi,sin_tht * sin_phi,cos_tht };
}

__forceinline__ __device__ bool       traceOccluded(const owl::RayT<RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT>& ray) {
  unsigned int occluded = 0;
  optixTrace(optixLaunchParams.tlas, { ray.origin.x,ray.origin.y,ray.origin.z }, { ray.direction.x,ray.direction.y,ray.direction.z }, 0.0f, 1e10f, 0.0f, 255u,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    RAY_TYPE_OCCLUDED,
    RAY_TYPE_COUNT,
    RAY_TYPE_OCCLUDED, occluded);
  return occluded;
}

extern "C" {
  __global__ void __raygen__default() {
    auto sbt_rg_data= reinterpret_cast<const SBTRaygenData*>(optixGetSbtDataPointer());
    auto launch_idx = optixGetLaunchIndex();
    auto res        = owl::vec3f(0.0f, 0.0f, 0.0f);
    auto frame_index = launch_idx.y * sbt_rg_data->width + launch_idx.x;

    owl::LCG<24> random = {};
    random.init(frame_index, sbt_rg_data->sample);

    Payload payload;
    payload.random = random;

    constexpr int num_samples = 10;
    for (int i = 0; i < num_samples; ++i) {
      auto uv = make_float2(
        ((float)launch_idx.x + payload.random()) / (float)sbt_rg_data->width,
        ((float)launch_idx.y + payload.random()) / (float)sbt_rg_data->height
      );

      uv.x = 2.0f * uv.x - 1.0f;
      uv.y = 2.0f * uv.y - 1.0f;

      auto org = sbt_rg_data->camera.eye;
      auto dir = (sbt_rg_data->camera.dir_w + uv.x * sbt_rg_data->camera.dir_u + uv.y * sbt_rg_data->camera.dir_v);

      dir = owl::normalize(dir);
      auto tmin = sbt_rg_data->camera.near_clip;
      auto tmax = sbt_rg_data->camera.far_clip;

      owl::RayT<RAY_TYPE_RADIANCE, RAY_TYPE_COUNT> ray(org, dir, tmin, tmax);

      payload.color = make_float3(0.0f, 0.0f, 0.0f);
      owl::traceRay(optixLaunchParams.tlas, ray, payload, OPTIX_RAY_FLAG_NONE);

      res += owl::vec3f(payload.color);
    }
    res /= num_samples;

    auto prv_accum                         = owl::vec3f(sbt_rg_data->accum_buffer[frame_index]);
    auto cur_accum                         = prv_accum + res;
    auto cur_frame                         = cur_accum / static_cast<float>(sbt_rg_data->sample +1);
    sbt_rg_data->accum_buffer[frame_index] = make_float3(cur_accum.x, cur_accum.y, cur_accum.z);
    sbt_rg_data->frame_buffer[frame_index] = make_float3(cur_frame.x, cur_frame.y, cur_frame.z);
  }
  __global__ void __miss__default()   {
    auto sbt_ms_data = reinterpret_cast<const SBTMissData*>(optixGetSbtDataPointer());
    auto& payload = owl::getPRD<Payload>();
    auto pos = owl::vec3f(optixGetWorldRayOrigin())+ optixGetRayTmax() * owl::vec3f(optixGetWorldRayDirection());
    payload.color = optixLaunchParams.light.envmap.sample(pos);
  }
  __global__ void __miss__occlude() {
    optixSetPayload_0(0);
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
    auto tmax          = optixGetRayTmax();
    auto position      = origin + tmax *direction;

    auto vertex0       = owl::vec3f(vertex_buffer[tri_index.x]);
    auto vertex1       = owl::vec3f(vertex_buffer[tri_index.y]);
    auto vertex2       = owl::vec3f(vertex_buffer[tri_index.z]);

    auto normal0       = owl::vec3f(normal_buffer[tri_index.x]);
    auto normal1       = owl::vec3f(normal_buffer[tri_index.y]);
    auto normal2       = owl::vec3f(normal_buffer[tri_index.z]);

    auto texcrd0       = owl::vec2f(texcrd_buffer[tri_index.x]);
    auto texcrd1       = owl::vec2f(texcrd_buffer[tri_index.y]);
    auto texcrd2       = owl::vec2f(texcrd_buffer[tri_index.z]);

    auto v_normal      = owl::normalize(owl::cross(vertex1-vertex0,vertex2-vertex1));
    auto s_normal      = owl::normalize(owl::vec3f((1.0f - barycentrics.x - barycentrics.y) * normal0 + barycentrics.x * normal1 + barycentrics.y * normal2));
    auto texcrd        = (1.0f - barycentrics.x - barycentrics.y) * texcrd0 + barycentrics.x * texcrd1 + barycentrics.y * texcrd2;
    auto cosine_in     = -owl::dot(direction, s_normal);
    //if (cosine_in < 0.0f) {
    //  s_normal *= -1.0f;
    //}

    auto& payload = owl::getPRD<Payload>();
    auto surface = optixLaunchParams.surfaces[sbt_hg_data->surfaces];

    if (surface.type & SURFACE_TYPE_DIFFUSE) { // DIFFUSEはおおむね一致。
      auto diffuse     = surface.loadDiffuse(optixLaunchParams.textures, texcrd.x, texcrd.y);
      Onb onb(s_normal);
      auto reflection  = onb.local(random_in_pdf_cosine(payload.random));

      owl::RayT<RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT> ray(position+0.01f*s_normal, reflection, 0.0f, 1000000.0f);

      auto refl_color = optixLaunchParams.light.envmap.sample(reflection);
      payload.color   = static_cast<float>(!traceOccluded(ray)) * diffuse.reflectance *refl_color;
    }
    else if (surface.type & SURFACE_TYPE_PLASTIC) { // PLASTICは一致まで時間かかる
      auto specular_reflection  = owl::normalize(direction + 2.0f * cosine_in * s_normal);

      auto plastic              = surface.loadPlastic(optixLaunchParams.textures, texcrd.x, texcrd.y);
      auto sine_in_2            = 1.0f - cosine_in * cosine_in;
      auto sine_out_2           = sine_in_2 / (plastic.eta * plastic.eta);
      auto r0                   = fresnel(plastic.eta, cosine_in * cosine_in, 1.0f - sine_out_2);
      auto t0                   = 1.0f - r0;

      auto diffuse_reflection   = random_in_pdf_cosine(payload.random);
      auto cos2_out             = diffuse_reflection.z    ;
      auto sin2_out_2           = 1.0f - cos2_out * cos2_out;
      auto sin2_in_2            = sin2_out_2 / (plastic.eta * plastic.eta);

      Onb onb(s_normal);
      diffuse_reflection        = onb.local(diffuse_reflection);
      auto r1                   = fresnel(1.0f/plastic.eta, 1.0f - sin2_in_2, cos2_out * cos2_out);
      auto t1                   = 1.0f - r1;

      if (payload.random() < 0.5f)
      {
        owl::RayT<RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT> ray(position + 0.01f * s_normal, specular_reflection, 0.0f, 1000000.0f);
        auto spec_refl_color                 = optixLaunchParams.light.envmap.sample(specular_reflection);
        payload.color                        = (!traceOccluded(ray)) * 2.0f * r0 * plastic.specular_reflectance * spec_refl_color;
      }
      else
      {
        owl::RayT<RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT> ray(position + 0.01f * s_normal, diffuse_reflection, 0.0f, 1000000.0f);
        bool nonlinear                       = plastic.int_fresnel_diffuse_reflectance > 0.0f;
        auto int_fresnel_diffuse_reflectance = fabsf(plastic.int_fresnel_diffuse_reflectance);
        auto diff_reflectance_fact           = nonlinear ? plastic.diffuse_reflectance : owl::vec3f(1.0f);
        auto diff_crr                        = owl::vec3f(1.0f) - (diff_reflectance_fact* int_fresnel_diffuse_reflectance);
        auto diff_refl_color                 = optixLaunchParams.light.envmap.sample(diffuse_reflection);
        payload.color                        = (!traceOccluded(ray)) * 2.0f * t0 * t1 *(plastic.diffuse_reflectance/ (diff_crr *plastic.eta*plastic.eta)) * diff_refl_color;
      }

    }
    //payload.color = 0.5f*(normal+owl::vec3f(1.0f));

  }
  __global__ void __closesthit__occlude_triangle() {
    optixSetPayload_0(1);
  }
}
