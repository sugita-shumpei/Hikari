
#include <optix_device.h>
#include <owl/owl_device.h>
#include <owl/common/owl-common.h>
#include <owl/common/math/random.h>
#include <owl/common/math/vec.h>
#include "deviceCode.h"

using Random = owl::PCG32;

#define FORMATTER_VEC3F(V) (V.x), (V.y), (V.z)


extern "C" { __constant__ LaunchParams optixLaunchParams;  }

struct PayloadData {
  owl::vec3f   s_normal;
  unsigned int surface_idx;
  owl::vec3f   g_normal;
  float        distance;
  owl::vec2f   texcoord;
};
__forceinline__ __device__ void       traceRadiance(const owl::RayT<0, 1>& ray, PayloadData& payload) {
  owl::trace(optixLaunchParams.tlas, ray, RAY_TYPE_COUNT, payload, RAY_TYPE_RADIANCE);
}
__forceinline__ __device__ bool       traceOccluded(const owl::RayT<0, 1>& ray) {
  unsigned int occluded = 0;
  optixTrace(optixLaunchParams.tlas, { ray.origin.x,ray.origin.y,ray.origin.z }, { ray.direction.x,ray.direction.y,ray.direction.z }, 0.0f, 1e10f, 0.0f, 255u,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    RAY_TYPE_OCCLUDED,
    RAY_TYPE_COUNT,
    RAY_TYPE_OCCLUDED, occluded);
  return occluded;
}
__forceinline__ __device__ float      fresnel1(float eta, float cos_in_2, float cos_out_2) {
  float cos_in = sqrtf(cos_in_2);
  float cos_out= sqrtf(cos_out_2);
  float rp = (eta * cos_in - cos_out) / (eta * cos_in + cos_out);
  float rs = (eta * cos_out - cos_in) / (eta * cos_out + cos_in);
  return 0.5f * (rp * rp + rs * rs);
}
__forceinline__ __device__ float      fresnel2(float eta, float k       , float cos_in_2, float cos_out_2) {

  float cos_in  = sqrtf(cos_in_2);
  float cos_out = sqrtf(cos_out_2);
  float rp1     = (eta * cos_in - cos_out);
  float rp2     = (eta * cos_in + cos_out);
  float rpk     = (k * cos_in);
  float rs1     = (eta * cos_out - cos_in);
  float rs2     = (eta * cos_out + cos_in);
  float rsk     = (k * cos_out);
  float rp      = (rp1*rp1 + rpk*rpk)/ (rp2 * rp2 + rpk * rpk);
  float rs      = (rs1*rs1 + rsk*rsk)/ (rs2 * rs2 + rsk * rsk);
  return 0.5f * (rp + rs);
}
__forceinline__ __device__ owl::vec3f fresnel2(const owl::vec3f&     eta, const owl::vec3f& k, float cos_in_2, const owl::vec3f& cos_out_2) {
  return owl::vec3f(fresnel2(eta.x, k.x, cos_in_2, cos_out_2.x), fresnel2(eta.y, k.y, cos_in_2, cos_out_2.y), fresnel2(eta.z, k.z, cos_in_2, cos_out_2.z));
}
__forceinline__ __device__ float      eval_ndf_isotropic_beckmann(float alpha_2, float n_cos_m_2, float xi_of_m_dot_n) {
  float n_sin_m_2 = fmaxf(1.0f - n_cos_m_2,0.0f);
  float n_cos_m_4 = n_cos_m_2 * n_cos_m_2;
  float n_tan_m_2 = n_sin_m_2 / n_cos_m_2;
  float log_res   = -n_tan_m_2 / alpha_2 - logf(M_PI * alpha_2 * n_cos_m_4);
  return xi_of_m_dot_n * expf(log_res);
}
__forceinline__ __device__ float      schlick_g1_isotropic_beckmann(float a,float xi_of_v_dot_m_per_v_dot_n) {
  if (a < 1.6f) { return xi_of_v_dot_m_per_v_dot_n * (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a); }
  else { return xi_of_v_dot_m_per_v_dot_n; }
}
__forceinline__ __device__ float      schlick_g2_isotropic_beckmann(float a_v, float a_l, float xi_of_v_dot_m_per_v_dot_n, float xi_of_l_dot_m_per_l_dot_n) {
  return schlick_g1_isotropic_beckmann(a_v, xi_of_v_dot_m_per_v_dot_n) * schlick_g1_isotropic_beckmann(a_l, xi_of_l_dot_m_per_l_dot_n);
}
// sample D(wm,wn) dot(wm,wn)
__forceinline__ __device__ owl::vec3f sample_pdf_isotropic_beckmann(float alpha_2, Random& random) {
  float tan_tht_m_2     =-alpha_2 * logf(1.0f - random());
  float inv_cos_tht_m_2 = 1.0f + tan_tht_m_2;
  float cos_tht_m_2     = 1.0f / inv_cos_tht_m_2;
  float sin_tht_m_2     = tan_tht_m_2 * cos_tht_m_2;
  float cos_tht_m       = sqrtf(cos_tht_m_2);
  float sin_tht_m       = sqrtf(sin_tht_m_2);
  float phi_m           = 2.0f * static_cast<float>(M_PI)*random();
  float cos_phi_m       = cosf(phi_m);
  float sin_phi_m       = sinf(phi_m);
  return owl::vec3f(sin_tht_m * cos_phi_m, sin_tht_m * sin_phi_m, cos_tht_m);
}
__forceinline__ __device__ owl::vec4f sample_and_eval_pdf_isotropic_beckmann(float alpha_2, Random& random) {
  float tan_tht_m_2     = -alpha_2 * logf(1.0f - random());
  float inv_cos_tht_m_2 = 1.0f + tan_tht_m_2;
  float cos_tht_m_2 = 1.0f / inv_cos_tht_m_2;
  float sin_tht_m_2 = tan_tht_m_2 * cos_tht_m_2;
  float cos_tht_m = sqrtf(cos_tht_m_2);
  float sin_tht_m = sqrtf(sin_tht_m_2);
  float phi_m = 2.0f * static_cast<float>(M_PI) * random();
  float cos_phi_m = cosf(phi_m);
  float sin_phi_m = sinf(phi_m);
  return owl::vec4f(sin_tht_m * cos_phi_m, sin_tht_m * sin_phi_m, cos_tht_m, eval_ndf_isotropic_beckmann(alpha_2,cos_tht_m_2,1.0f)* cos_tht_m);
}
__forceinline__ __device__ float      eval_pdf_isotropic_beckmann(float alpha_2, float n_cos_m) {
  return eval_ndf_isotropic_beckmann(alpha_2, n_cos_m * n_cos_m, static_cast<float>(n_cos_m > 0.0f)) * n_cos_m ;
}
__forceinline__ __device__ owl::vec3f sample_pdf_cosine(Random& random) {
  float cos_tht = sqrtf(1 - random());
  float sin_tht = sqrtf(fmaxf(1 - cos_tht * cos_tht,0.0f));
  float phi = 2.0f * M_PI * random();
  float cos_phi = cosf(phi);
  float sin_phi = sinf(phi);
  return { sin_tht * cos_phi,sin_tht * sin_phi,cos_tht };
}
__forceinline__ __device__ float      eval_pdf_cosine(float cos_tht) {
  return fmaxf(cos_tht,0.0f)*M_1_PI;
}
__forceinline__ __device__ owl::vec3f sample_and_eval_bsdf_isotropic_beckmann(const owl::vec3f& eta, const owl::vec3f& k, float alpha, const owl::vec3f& wi, owl::vec3f& wm, owl::vec3f& weight, float& pdf, Random& random) {
  float alpha_2         = alpha * alpha;
  owl::vec4f wm_and_pdf = sample_and_eval_pdf_isotropic_beckmann(alpha_2, random);
  wm              = owl::vec3f(wm_and_pdf);
  pdf             = wm_and_pdf.w;
  float n_cos_m   = wm.z;
  float n_cos_i   = wi.z;
  float m_cos_i   = owl::dot(wi, wm);
  auto  wo        = owl::normalize(2.0f * m_cos_i * wm - wi);
  float n_cos_o   = wo.z;
  float m_cos_o   = owl::dot(wo, wm);
  float n_cos_i_2 = n_cos_i * n_cos_i;
  float n_cos_o_2 = n_cos_o * n_cos_o;
  float n_sin_i_2 = fmaxf(1.0f-n_cos_i_2,0.0f);
  float n_sin_o_2 = fmaxf(1.0f-n_cos_o_2,0.0f);
  float n_sin_i   = sqrtf(n_sin_i_2);
  float n_sin_o   = sqrtf(n_sin_o_2);
  float a_i       = n_sin_i!=0.0f?n_cos_i/(alpha*n_sin_i):copysignf(FLT_MAX,n_cos_i);
  float a_o       = n_sin_o!=0.0f?n_cos_o/(alpha*n_sin_o):copysignf(FLT_MAX,n_cos_o);
  float xi_of_i_dot_m_per_i_dot_n = static_cast<float>((n_cos_i * m_cos_i) > 0.0f);
  float xi_of_o_dot_m_per_o_dot_n = static_cast<float>((n_cos_o * m_cos_o) > 0.0f);
  float g2        = schlick_g2_isotropic_beckmann(a_i,a_o,xi_of_i_dot_m_per_i_dot_n,xi_of_o_dot_m_per_o_dot_n);
  float m_sin_i_2 = fmaxf(1.0f - m_cos_i * m_cos_i,0.0f);
  auto  m_sin_t_2 = m_sin_i_2 / (eta * eta);
  auto  m_cos_t_2 = owl::vec3f(fmaxf(1.0f - m_sin_t_2.x, 0.0f), fmaxf(1.0f - m_sin_t_2.y, 0.0f), fmaxf(1.0f - m_sin_t_2.z, 0.0f));
  auto  f         = fresnel2(eta, k, m_cos_i * m_cos_i, m_cos_t_2);
  weight          = f * g2 * fabsf(m_cos_i) / (n_cos_i* n_cos_m);
  if (n_cos_i <= 0.0f) { weight = 0.0f; }
  if (n_cos_o <= 0.0f) { weight = 0.0f; }
  return wo;
}
__forceinline__ __device__ bool       shade_material(
  const PayloadData& payload   ,
  float              min_depth ,
  float              max_depth ,
  owl::vec3f&        ray_org   ,
  owl::vec3f&        ray_dir   ,
  owl::vec3f&        color     ,
  owl::vec3f&        throughput,
  Random&      random    ) {
  ray_org           = ray_org + payload.distance * ray_dir;
  auto s_normal     = payload.s_normal;
  auto g_normal     = payload.g_normal;
  auto s_cosine_in  = -owl::dot(s_normal, ray_dir);
  auto g_cosine_in  = -owl::dot(g_normal, ray_dir);

  // CLOSEST HIT
  if (payload.surface_idx > 0) {
    auto surface_idx = payload.surface_idx - 1;
    // Surfaceを取得
    auto& surface    = optixLaunchParams.surfaces[surface_idx];
    // Light
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_DIFFUSE   ) { // DIFFUSEはおおむね一致。
      if (s_cosine_in < 0.0f || g_cosine_in < 0.0f) { return true; }
      Onb  onb(s_normal);
      auto refl_dir      = onb.local_to_world(sample_pdf_cosine(random));
      float g_cosine_out = owl::dot(g_normal, refl_dir);
      if   (g_cosine_out < 0.0f) { return true; }

      auto diffuse       = surface.loadDiffuse(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);
      ray_org            = offset_ray(ray_org, s_normal);
      ray_dir            = refl_dir;
      throughput        *= diffuse.reflectance;
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_CONDUCTOR ) { // PLASTICはおおむね一致。
      if (s_cosine_in < 0.0f || g_cosine_in < 0.0f) { return true; }
      auto refl_dir     = owl::normalize(ray_dir + 2.0f * s_cosine_in * s_normal);
      float g_cosine_out= owl::dot(g_normal, refl_dir);
      if (g_cosine_out < 0.0f) { return true; }

      auto conductor    = surface.loadConductor(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);

      auto cos_1_in     = s_cosine_in;
      auto cos_1_in_sq  = cos_1_in * cos_1_in;
      auto sin_1_in_sq  = 1.0f - cos_1_in_sq;
      auto sin_1_out_sq = owl::vec3f(sin_1_in_sq) / (conductor.eta * conductor.eta);
      auto cos_1_out_sq = owl::vec3f(fmaxf(1.0f - sin_1_out_sq.x, 0.0f), fmaxf(1.0f - sin_1_out_sq.y, 0.0f), fmaxf(1.0f - sin_1_out_sq.z, 0.0f));
      auto r0           = fresnel2(conductor.eta, conductor.k, cos_1_in_sq, cos_1_out_sq);
      
      ray_org           = offset_ray(ray_org, s_normal);
      ray_dir           = refl_dir;
      throughput       *= r0 * conductor.specular_reflectance;
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_DIELECTRIC) { // PLASTICはおおむね一致。
      //if (s_cosine_in* g_cosine_in < 0.0f) { return true; }
      auto  dielectric  = surface.loadDielectric(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);
      float eta         = s_cosine_in > 0.0f ? dielectric.eta : 1.0f / dielectric.eta;
      auto  r_normal    = s_cosine_in > 0.0f ? s_normal : -s_normal;

      auto cos_1_in     = s_cosine_in;
      auto cos_1_in_sq  = cos_1_in * cos_1_in;
      auto sin_1_in_sq  = 1.0f - cos_1_in_sq;
      auto sin_1_out_sq = sin_1_in_sq / (eta* eta);
      auto cos_1_out_sq = fmaxf(1.0f - sin_1_out_sq, 0.0f);
      auto r0           = fresnel1(eta, cos_1_in_sq, cos_1_out_sq);

      if (random() < r0) {
        auto refl_dir   = owl::normalize(ray_dir + 2.0f * s_cosine_in * s_normal);
        ray_org         = offset_ray(ray_org,r_normal);
        ray_dir         = refl_dir;
        throughput     *= dielectric.specular_reflectance;
      }
      else {
        auto tran_dir  = owl::normalize((ray_dir + s_cosine_in * s_normal) / eta - sqrtf(cos_1_out_sq) * r_normal);
        ray_org        = offset_ray(ray_org,-r_normal);
        ray_dir        = tran_dir;
        throughput    *= dielectric.specular_transmittance/(eta*eta);
      }
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_THIN_DIELECTRIC) { // PLASTICはおおむね一致。
      //if (s_cosine_in* g_cosine_in < 0.0f) { return true; }
      auto  thin            = surface.loadThinDielectric(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);
      float eta             = thin.dielectric.eta;
      auto  r_normal        = s_cosine_in > 0.0f ? s_normal : -s_normal;

      auto cos_1_in         = s_cosine_in;
      auto cos_1_in_sq      = cos_1_in * cos_1_in;
      auto sin_1_in_sq      = 1.0f - cos_1_in_sq;
      auto sin_1_out_sq     = sin_1_in_sq / (eta * eta);
      auto cos_1_out_sq     = fmaxf(1.0f - sin_1_out_sq, 0.0f);
      auto r                = fresnel1(eta, cos_1_in_sq, cos_1_out_sq);
      auto t                = 1.0f - r;
      if (r < 1.0f) {
        r += t * t * r / (1.0f - r * r);
      }

      if (random() < r) {
        auto refl_dir = owl::normalize(ray_dir + 2.0f * s_cosine_in * s_normal);
        ray_org       = offset_ray(ray_org, r_normal);
        ray_dir = refl_dir;
        throughput *= thin.dielectric.specular_reflectance;
      }
      else {
        auto tran_dir = ray_dir;
        ray_org  = offset_ray(ray_org, -r_normal);
        ray_dir = tran_dir;
        throughput *= thin.dielectric.specular_transmittance;
      }
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_PLASTIC   ) { // PLASTICはおおむね一致。
      if (s_cosine_in < 0.0f || g_cosine_in < 0.0f) { return true; }
      auto plastic                         = surface.loadPlastic(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);
      bool nonlinear                       = plastic.int_fresnel_diffuse_reflectance > 0.0f;
      auto int_fresnel_diffuse_reflectance = fabsf(plastic.int_fresnel_diffuse_reflectance);
      auto diff_reflectance_fact           = nonlinear ? plastic.diffuse_reflectance : owl::vec3f(1.0f);
      auto diff_crr                        = owl::vec3f(1.0f) - (diff_reflectance_fact * int_fresnel_diffuse_reflectance);

      auto cos_1_in     = s_cosine_in;
      auto cos_1_in_sq  = cos_1_in * cos_1_in;
      auto sin_1_in_sq  = 1.0f - cos_1_in_sq;
      auto sin_1_out_sq = sin_1_in_sq / (plastic.eta * plastic.eta);
      auto cos_1_out_sq = fmaxf(1.0f - sin_1_out_sq, 0.0f);

      auto r0 = fresnel1(plastic.eta, cos_1_in_sq, cos_1_out_sq);
      auto t0 = 1.0f - r0;

      auto spec_refl_dir = owl::normalize(ray_dir + 2.0f * s_cosine_in * s_normal);
      auto diff_refl_dir = sample_pdf_cosine(random);

      auto cos_2_in     = diff_refl_dir.z;
      auto cos_2_in_sq  = cos_2_in * cos_2_in;
      auto sin_2_in_sq  = 1.0f - cos_2_in_sq;
      auto sin_2_out_sq = sin_2_in_sq / (plastic.eta * plastic.eta);
      auto cos_2_out_sq = fmaxf(1.0f - sin_2_out_sq, 0.0f);

      Onb onb(s_normal);
      diff_refl_dir = onb.local_to_world(diff_refl_dir);

      auto r1 = fresnel1(plastic.eta, cos_2_in_sq, cos_2_out_sq);
      auto t1 = 1.0f - r1;

      auto spec_g_cos_out = owl::dot(spec_refl_dir, g_normal);
      auto diff_g_cos_out = owl::dot(diff_refl_dir, g_normal);

      auto total_spec_reflectance =  r0 * static_cast<float>(spec_g_cos_out > 0.0f) * plastic.specular_reflectance;
      auto total_diff_reflectance = (t0 * t1 / (plastic.eta * plastic.eta)) * static_cast<float>(diff_g_cos_out > 0.0f) * (plastic.diffuse_reflectance / diff_crr);

      auto ave_total_spec_reflectance = owl::dot(total_spec_reflectance, owl::vec3f(1.0f)) / 3.0f;
      auto ave_total_diff_reflectance = owl::dot(total_diff_reflectance, owl::vec3f(1.0f)) / 3.0f;

      auto sum_ave_reflectance = (ave_total_spec_reflectance + ave_total_diff_reflectance);
      if ( sum_ave_reflectance <= 0.0f) { return true; }

      auto prob = (ave_total_spec_reflectance) / sum_ave_reflectance;

      ray_org      += 0.01f * s_normal;
      if (random() < prob)
      {
        ray_dir     = spec_refl_dir;
        throughput *= total_spec_reflectance/prob;
      }
      else
      {
        ray_dir     = diff_refl_dir;
        throughput *= total_diff_reflectance/(1.0f-prob);
      }
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_ROUGH_CONDUCTOR_ISOTROPIC) { // PLASTICはおおむね一致。
      if (s_cosine_in < 0.0f || g_cosine_in < 0.0f) { return true; }
      auto rough_isotropic = surface.loadRoughConductorIsotropic(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);
      Onb onb(s_normal);
      auto wi     = onb.world_to_local(-ray_dir);
      auto wm     = owl::vec3f();
      auto weight = owl::vec3f();
      auto pdf    = 0.0f;
      auto wo     = sample_and_eval_bsdf_isotropic_beckmann(rough_isotropic.conductor.eta, rough_isotropic.conductor.k, rough_isotropic.alpha, wi, wm, weight,pdf, random);
      auto avg_wei = weight.x + weight.y + weight.z;
      if (avg_wei <= 0.0f) { return true; }
      ray_org     = offset_ray(ray_org, s_normal);
      ray_dir     = onb.local_to_world(wo);
      throughput *= weight * rough_isotropic.conductor.specular_reflectance;
      return false;
    }
    if ((surface.type & SURFACE_TYPE_MASK_ALL) == SURFACE_TYPE_ROUGH_CONDUCTOR_ANISOTROPIC) { // PLASTICはおおむね一致。
      if (s_cosine_in < 0.0f || g_cosine_in < 0.0f) { return true; }
      auto refl_dir = owl::normalize(ray_dir + 2.0f * s_cosine_in * s_normal);
      float g_cosine_out = owl::dot(g_normal, refl_dir);
      if (g_cosine_out < 0.0f) { return true; }

      auto rough_anisotropic = surface.loadRoughConductorAnisotropic(optixLaunchParams.textures, payload.texcoord.x, payload.texcoord.y);

      auto cos_1_in = s_cosine_in;
      auto cos_1_in_sq = cos_1_in * cos_1_in;
      auto sin_1_in_sq = 1.0f - cos_1_in_sq;
      auto sin_1_out_sq = owl::vec3f(sin_1_in_sq) / (rough_anisotropic.conductor.eta * rough_anisotropic.conductor.eta);
      auto cos_1_out_sq = owl::vec3f(fmaxf(1.0f - sin_1_out_sq.x, 0.0f), fmaxf(1.0f - sin_1_out_sq.y, 0.0f), fmaxf(1.0f - sin_1_out_sq.z, 0.0f));
      auto r0 = fresnel2(rough_anisotropic.conductor.eta, rough_anisotropic.conductor.k, cos_1_in_sq, cos_1_out_sq);

      ray_org = offset_ray(ray_org, s_normal);
      ray_dir = refl_dir;
      throughput *= r0 * rough_anisotropic.conductor.specular_reflectance;
      return false;
    }
    return false;
  }
  // MISS
  else
  {
    color += throughput * optixLaunchParams.light.envmap.sample(ray_dir);
    return true;
  }
}
// 実際の描画処理はここで実行
OPTIX_RAYGEN_PROGRAM(default)() {
  const owl::vec2i idx = owl::getLaunchIndex();
  const owl::vec2i dim = owl::getLaunchDims();
  const SBTRaygenData& sbt_rg_data = owl::getProgramData<SBTRaygenData>();

  auto frame_index = dim.x * idx.y + idx.x;

  constexpr auto frame_samples = 1;
  constexpr auto trace_depth   = 2;

  auto payload = PayloadData();
  Random random = {};
  random.init(frame_index, sbt_rg_data.sample);

  auto color = owl::vec3f(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < frame_samples; ++i) {
    payload = PayloadData();
    auto uv = owl::vec2f(
      2.0f * (((float)idx.x + random()) / (float)sbt_rg_data.width )-1.0f,
      2.0f * (((float)idx.y + random()) / (float)sbt_rg_data.height)-1.0f
    );

    auto ray_org    = sbt_rg_data.camera.eye;
    auto ray_dir    = (sbt_rg_data.camera.dir_w + uv.x * sbt_rg_data.camera.dir_u + uv.y * sbt_rg_data.camera.dir_v);

    auto throughput = owl::vec3f(1.0f, 1.0f, 1.0f);
    bool done       = false;
    for (int j = 0; (j < trace_depth) && !done; ++j) {
      float tmin = (j == 0) ? sbt_rg_data.camera.near_clip : 0.0f;
      float tmax = (j == 0) ? sbt_rg_data.camera.far_clip  : 1e11f;
      owl::RayT<0, 1> ray(ray_org, ray_dir, tmin, tmax);
      traceRadiance(ray, payload);
      done   = shade_material(payload, 0.01f, 1e11f, ray_org, ray_dir, color, throughput, random);

    }
  }

  auto prv_accum = owl::vec3f(sbt_rg_data.accum_buffer[frame_index]);
  auto cur_accum = prv_accum + color;
  auto cur_frame = cur_accum / static_cast<float>(sbt_rg_data.sample + frame_samples);
  sbt_rg_data.accum_buffer[frame_index] = make_float3(cur_accum.x, cur_accum.y, cur_accum.z);
  sbt_rg_data.frame_buffer[frame_index] = make_float3(cur_frame.x, cur_frame.y, cur_frame.z);
}
// レイタイプ: Radiance
// 最近傍シェーダ(サーフェス情報を取得)
OPTIX_CLOSEST_HIT_PROGRAM(default_triangle)() {
  const SBTHitgroupData& sbt_hg_data = owl::getProgramData<SBTHitgroupData>();
  PayloadData&     payload = owl::getPRD<PayloadData>();
  auto vertex_buffer = reinterpret_cast<const float3*>(sbt_hg_data.vertex_buffer);
  auto normal_buffer = sbt_hg_data.normal_buffer;
  auto texcrd_buffer = sbt_hg_data.texcrd_buffer;
  auto index_buffer  = reinterpret_cast<const uint3*>(sbt_hg_data.index_buffer);
  auto prim_index    = optixGetPrimitiveIndex();
  auto tri_index     = index_buffer[prim_index];
  auto v0 = owl::vec3f(vertex_buffer[tri_index.x]);
  auto v1 = owl::vec3f(vertex_buffer[tri_index.y]);
  auto v2 = owl::vec3f(vertex_buffer[tri_index.z]);

  auto n0 = owl::vec3f(normal_buffer[tri_index.x]);
  auto n1 = owl::vec3f(normal_buffer[tri_index.y]);
  auto n2 = owl::vec3f(normal_buffer[tri_index.z]);

  auto t0       = owl::vec2f(texcrd_buffer[tri_index.x]);
  auto t1       = owl::vec2f(texcrd_buffer[tri_index.y]);
  auto t2       = owl::vec2f(texcrd_buffer[tri_index.z]);
  auto v01      = v1 - v0;
  auto v12      = v2 - v1;
  float2 bary   = optixGetTriangleBarycentrics();
  auto f_normal = owl::normalize(owl::cross(v01, v12));
  auto s_normal = owl::normalize((1.0f - (bary.x + bary.y)) * n0 + bary.x * n1 + bary.y * n2);

  auto vt             = (1.0f - (bary.x + bary.y)) * t0 + bary.x * t1 + bary.y * t2;
  payload.surface_idx = sbt_hg_data.surfaces + 1;
  payload.texcoord    = vt;
  payload.distance    = optixGetRayTmax();
  payload.g_normal    = f_normal;
  payload.s_normal    = s_normal;
  // 実際の処理はCallableで実行する
}
// ミスシェーダ  (サーフェス情報を取得)
OPTIX_MISS_PROGRAM(default)() {
  PayloadData& payload = owl::getPRD<PayloadData>();
  payload.surface_idx  = 0;
  payload.texcoord     = { 0.0f, 0.0f };
  payload.distance     = 0.0f;
  payload.g_normal     = { 0.0f,0.0f, 0.0f };
  payload.s_normal     = { 0.0f,0.0f, 0.0f };
}
// レイタイプ: Occluded
// 最近傍シェーダ(可視情報を取得)
OPTIX_CLOSEST_HIT_PROGRAM(occlude_triangle)() {
  optixSetPayload_0(1);
}
// ミスシェーダ  (可視情報を取得)
OPTIX_MISS_PROGRAM(occlude)() {
  optixSetPayload_0(0);
}
//// AnyHitシェーダ(αテストを起動）
//OPTIX_ANY_HIT_PROGRAM(simpleAH)() {
//  auto& ch_data = owl::getProgramData<HitgroupData>();
//  if (optixIsTriangleHit()) {
//    auto pri_idx = optixGetPrimitiveIndex();
//    auto tri_idx = ch_data.indices[pri_idx];
//    auto bary = optixGetTriangleBarycentrics();
//
//    auto vt0 = ch_data.uvs[tri_idx.x];
//    auto vt1 = ch_data.uvs[tri_idx.y];
//    auto vt2 = ch_data.uvs[tri_idx.z];
//
//    auto vt = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
//    auto tmp_col = tex2D<float4>(ch_data.texture_alpha, vt.x, vt.y);
//    if (tmp_col.w * tmp_col.x * tmp_col.y * tmp_col.z < 0.5f) {
//      optixIgnoreIntersection();
//    }
//  }
//}
