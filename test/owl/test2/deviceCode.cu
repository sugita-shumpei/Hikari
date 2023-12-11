#include <optix_device.h>
#include <owl/owl_device.h>
#include <owl/common/owl-common.h>
#include <owl/common/math/random.h>
#include <owl/common/math/vec.h>
#include "deviceCode.h"
extern "C" { __constant__ LaunchParams optixLaunchParams; }
struct PayloadData {
	owl::vec3f   s_normal;
	unsigned int mat_idx ;
	owl::vec3f   g_normal;
	float        distance;
	owl::vec2f   texcoord;
};
__forceinline__ __device__ void       traceRadiance(const owl::RayT<0, 1>& ray, PayloadData& payload) {
	owl::trace(optixLaunchParams.world, ray, RAY_TYPE_COUNT, payload, RAY_TYPE_RADIANCE);
}
__forceinline__ __device__ bool       traceOccluded(const owl::RayT<0,1>& ray) {
	unsigned int occluded = 0;
	optixTrace(optixLaunchParams.world, { ray.origin.x,ray.origin.y,ray.origin.z }, {ray.direction.x,ray.direction.y,ray.direction.z}, 0.0f, 1e10f, 0.0f, 255u,
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		RAY_TYPE_OCCLUDED,
		RAY_TYPE_COUNT,
		RAY_TYPE_OCCLUDED,occluded);
	return occluded;
}
__forceinline__ __device__ owl::vec2f normalize_uv(owl::vec2f vt) {
	vt.x = fmodf(vt.x, 1.0f);
	vt.y = fmodf(vt.y, 1.0f);
	if (vt.x < 0.0f) {
		vt.x = vt.x + 1.0f;
	}
	if (vt.y < 0.0f) {
		vt.y = vt.y + 1.0f;
	}
	return vt;
}
__forceinline__ __device__ float3     sample_sphere_map(const cudaTextureObject_t& tex, const float3& dir)
{
	float phi   = atan2f(dir.z, dir.x);
	float theta = dir.y;
	float x     = (phi / M_PI + 1.0f) * 0.5f;
	float y     = (dir.y + 1.0f) * 0.5f;
	auto res    = tex2D<float4>(tex, x, y);
	return make_float3(res.x, res.y, res.z);
}
__forceinline__ __device__ owl::vec3f eval_bsdf_phong(owl::vec3f ambient, owl::vec3f specular, float shininess, float light_cosine)
{
	return (ambient + specular * (0.5f * shininess + 1.0f) * powf(light_cosine, shininess)) / ((float)M_PI);
}
__forceinline__ __device__ float      eval_pdf_phong (float shininess, float light_cosine){
	return (0.5f * shininess + 0.5f) * powf(light_cosine, shininess) / M_PI;
}
__forceinline__ __device__ float      eval_pdf_cosine(float cosine) {
	return cosine / M_PI;
}
__forceinline__ __device__ owl::vec3f random_in_pdf_phong(float shininess, owl::LCG<24>& random) {
	float cos_tht = powf(random(),1.0f/(shininess+1.0f));
	float sin_tht = sqrtf(fmaxf(1 - cos_tht * cos_tht, 0.0f));
	float phi     = 2.0f * M_PI * random();
	float cos_phi = cosf(phi);
	float sin_phi = sinf(phi);
	return { sin_tht * cos_phi,sin_tht * sin_phi,cos_tht };
}
__forceinline__ __device__ owl::vec3f random_in_pdf_cosine(owl::LCG<24>& random) {
	float cos_tht   = (1 - random());
	float sin_tht   = sqrtf(fmaxf(1 - cos_tht* cos_tht,0.0f));
	float phi       = 2.0f * M_PI * random();
	float cos_phi   = cosf(phi);
	float sin_phi   = sinf(phi);
	return { sin_tht * cos_phi,sin_tht * sin_phi,cos_tht };
}
__forceinline__ __device__ float      fresnel(float f0, float geom_cosine) {
	return f0 + (1.0f - f0) * (1.0f - geom_cosine) * (1.0f - geom_cosine) * (1.0f - geom_cosine) * (1.0f - geom_cosine) * (1.0f - geom_cosine);
}
__forceinline__ __device__ bool       shade_material(
	const PayloadData& payload,
	float              min_depth,
	float              max_depth,
	owl::vec3f&        ray_org,
	owl::vec3f&        ray_dir,
	owl::vec3f&        color,
	owl::vec3f&        throughput,
	owl::LCG<24>&      random) {
	bool done = false;
	ray_org = ray_org + payload.distance * ray_dir;
	auto normal   = payload.s_normal;
	auto n_cosine = -owl::dot(normal, ray_dir);
	auto refl_dir = owl::normalize(ray_dir + 2.0f * n_cosine * normal);
	// CLOSEST HIT
	if (payload.mat_idx > 0) {
		// Materialを取得
		auto& material = optixLaunchParams.material_buffer[payload.mat_idx - 1];
		// Light
		if (material.material_type      == MATERIAL_TYPE_LIGHT) {
			ray_org += 0.01f * normal;
			auto material_light = MaterialParamsLight();
			material_light.set(material);
			if (material_light.texid > 0) {
				material_light.color *= owl::vec3f(tex2D<float4>(optixLaunchParams.texture_buffer[material_light.texid - 1], payload.texcoord.x, payload.texcoord.y));
			}
			color += throughput * material_light.color * fmaxf(n_cosine / M_PI, 0.0f);
			done = true;
		}
		// Delta Specular
		else if (material.material_type == MATERIAL_TYPE_SPECULAR_DELTA) {
			ray_org += 0.01f * normal;
			auto material_specular = MaterialParamsSpecular();
			material_specular.set(material);
			throughput *= material_specular.color;
			ray_dir = refl_dir;
		}
		// Delta Dielectric
		else if (material.material_type == MATERIAL_TYPE_DIELECTRIC_DELTA) {
			auto material_dielectric = MaterialParamsDielectric();
			material_dielectric.set(material);
			// 屈折率
			float ior = material_dielectric.ior;
			if (n_cosine < 0.0f) {
				normal   = -normal;
				n_cosine = -n_cosine;
				ior = 1.0f / ior;
			}
			auto n_cosine2 = n_cosine * n_cosine;
			auto n_sine2 = fmaxf(1.0f - n_cosine2, 0.0f);
			// 1.0 * sine_tht1 = n * sine_tht2;
			auto n_refr_sine2 = n_sine2 / (ior * ior);
			auto n_refr_cosine2 = 1.0f - n_refr_sine2;
			auto fr = 0.0f;
			if (n_refr_cosine2 < 0.0f) {
				fr = 1.0f;
			}
			else {
				auto n_refr_cosine = sqrtf(n_refr_cosine2);
				auto r_p = (ior * n_cosine - n_refr_cosine) / (ior * n_cosine + n_refr_cosine);
				auto r_s = (n_cosine - ior * n_refr_cosine) / (n_cosine + ior * n_refr_cosine);
				fr = (r_p * r_p + r_s * r_s) * 0.5f;
			}
			if (random() < fr) {
				ray_org += 0.01f * normal;
				ray_dir = refl_dir;
			}
			else {
				ray_org -= 0.01f * normal;
				auto refr_dir = owl::normalize((1.0f / ior) * ray_dir + ((1.0f / ior) * n_cosine - sqrtf(n_refr_cosine2)) * normal);
				ray_dir = refr_dir;
			}

		}
		// Phong 
		else if (material.material_type == MATERIAL_TYPE_SPECULAR_ROUGH_PHONG) {
			ray_org += +0.01f * normal;
			auto material_specular = MaterialParamsSpecular();
			material_specular.set(material);
			if (material_specular.texid_color > 0) {
				material_specular.color *= owl::vec3f(tex2D<float4>(optixLaunchParams.texture_buffer[material_specular.texid_color - 1], payload.texcoord.x, payload.texcoord.y));
			}
			if (material_specular.texid_factor1 > 0) {
				material_specular.factor1 *= (tex2D<float4>(optixLaunchParams.texture_buffer[material_specular.texid_factor1 - 1], payload.texcoord.x, payload.texcoord.y)).x;
			}

			if (optixLaunchParams.light_parallel.active) {
				owl::RayT<0, 1> shadow_ray(ray_org, optixLaunchParams.light_parallel.direction, min_depth, max_depth);
				if (!traceOccluded(shadow_ray)) {
					auto bsdf = eval_bsdf_phong({ 0.0f,0.0f,0.0f }, material_specular.color, material_specular.factor1, fmaxf(owl::dot(refl_dir, optixLaunchParams.light_parallel.direction), 0.0f));
					color += throughput * bsdf * fmaxf(owl::dot(normal, optixLaunchParams.light_parallel.direction), 0.0f) * optixLaunchParams.light_parallel.color;
				}
			}

			Onb onb1(refl_dir);
			auto direction = onb1.local(random_in_pdf_phong(material_specular.factor1, random));
			auto bsdf = eval_bsdf_phong({ 0.0f,0.0f,0.0f }, material_specular.color, material_specular.factor1, fmaxf(owl::dot(refl_dir, direction), 0.0f));
			if (owl::dot(normal, direction) > 0.0f) {
				throughput *= (owl::vec3f(M_PI) * bsdf);
			}
			else {
				done = true;
			}
			ray_dir = direction;
		}
		// Legacy Phong Composite
		else if (material.material_type == MATERIAL_TYPE_LEGACY_PHONG_COMPOSITE) {
			ray_org += +0.01f * normal;
			auto material_legacy_phong_composite = MaterialParamsLagacyPhongComposite();
			
			material_legacy_phong_composite.set(material);
			if (material_legacy_phong_composite.texid_color_ambient  > 0) {
				material_legacy_phong_composite.color_ambient *= owl::vec3f(tex2D<float4>(optixLaunchParams.texture_buffer[material_legacy_phong_composite.texid_color_ambient - 1], payload.texcoord.x, payload.texcoord.y));
			}
			if (material_legacy_phong_composite.texid_color_specular > 0) {
				material_legacy_phong_composite.color_specular *= owl::vec3f(tex2D<float4>(optixLaunchParams.texture_buffer[material_legacy_phong_composite.texid_color_specular - 1], payload.texcoord.x, payload.texcoord.y));
			}
			if (optixLaunchParams.light_parallel.active) {
				owl::RayT<0, 1> shadow_ray(ray_org, optixLaunchParams.light_parallel.direction,  min_depth, max_depth);
				if (!traceOccluded(shadow_ray)) {
					auto bsdf = eval_bsdf_phong(material_legacy_phong_composite.color_ambient, material_legacy_phong_composite.color_specular, material_legacy_phong_composite.shininess, fmaxf(owl::dot(refl_dir, optixLaunchParams.light_parallel.direction), 0.0f));
					color += throughput * bsdf * fmaxf(owl::dot(normal, optixLaunchParams.light_parallel.direction), 0.0f) * optixLaunchParams.light_parallel.color;
				}
			}

			Onb onb1(normal);
			auto direction1 = onb1.local(random_in_pdf_cosine(random));
			auto n_cosine1  = fmaxf(owl::dot(normal  , direction1), 0.0f);
			auto l_cosine1  = fmaxf(owl::dot(refl_dir, direction1), 0.0f);
			auto pdf11 = eval_pdf_cosine(n_cosine1);
			auto pdf12 = eval_pdf_phong(material_legacy_phong_composite.shininess, l_cosine1);

			Onb onb2(refl_dir);
			auto direction2 = onb2.local(random_in_pdf_phong(material_legacy_phong_composite.shininess, random));
			auto n_cosine2  = fmaxf(owl::dot(normal  , direction2), 0.0f);
			auto l_cosine2  = fmaxf(owl::dot(refl_dir, direction2), 0.0f);
			auto pdf21      = eval_pdf_cosine(n_cosine2);
			auto pdf22      = eval_pdf_phong(material_legacy_phong_composite.shininess, l_cosine2);

			auto kd         = owl::dot(material_legacy_phong_composite.color_ambient , owl::vec3f(1.0f / 3.0f));
			auto ks         = owl::dot(material_legacy_phong_composite.color_specular, owl::vec3f(1.0f / 3.0f));
			auto prob       = (kd + ks) > 0.0f ? kd / (kd + ks) : 0.5f;

			auto weight    = 0.0f;
			auto pdf       = 0.0f;
			auto n_cosine  = 0.0f;
			auto l_cosine  = 0.0f;
			auto direction = owl::vec3f();

			if (random() < prob) {
				weight = prob * pdf11 / (prob * pdf11 + (1.0f - prob) * pdf12);
				pdf = prob * pdf11;
				n_cosine = n_cosine1;
				l_cosine = l_cosine1;
				direction = direction1;
			}
			else {
				weight = (1.0f - prob) * pdf22 / (prob * pdf21 + (1.0f - prob) * pdf22);
				pdf = (1.0f - prob) * pdf22;
				n_cosine = n_cosine2;
				l_cosine = l_cosine2;
				direction = direction2;
			}

			auto bsdf = eval_bsdf_phong(material_legacy_phong_composite.color_ambient, material_legacy_phong_composite.color_specular, material_legacy_phong_composite.shininess, l_cosine);
			if (n_cosine > 0.0f) {
				throughput *= (bsdf * n_cosine * weight / pdf);
			}
			else {
				done = true;
			}
			ray_dir = direction;
		}
		// None
		else {
			done = true;
		}
	}
	// MISS
	else
	{
		color += throughput * optixLaunchParams.light_intensity * owl::vec3f(sample_sphere_map(optixLaunchParams.light_envmap, { ray_dir.x,ray_dir.y,ray_dir.z }));
		done = true;
	}
	return done;
}
// 実際の描画処理はここで実行
OPTIX_RAYGEN_PROGRAM(simpleRG)() {
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	auto& rg_data        = owl::getProgramData<RayGenData>();

	constexpr auto frame_samples = 2;
	constexpr auto trace_depth   = 8;

	auto payload = PayloadData();
	owl::LCG<24> random = {};
	random.init(dim.x * idx.y + idx.x, optixLaunchParams.accum_sample);

	auto color      = owl::vec3f(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < frame_samples; ++i) {
		payload         = PayloadData();
		float px        = ((float)idx.x + random() - 0.5f) / ((float)dim.x);
		float py        = ((float)idx.y + random() - 0.5f) / ((float)dim.y);
		auto ray_org    = rg_data.camera.eye;
		auto ray_dir    = owl::normalize(rg_data.camera.getRayDirection(px, py));
		auto throughput = owl::vec3f(1.0f, 1.0f, 1.0f);
		bool done       = false;
		for (int j = 0; (j < trace_depth) && !done; ++j) {
			owl::RayT<0, 1> ray(ray_org, ray_dir, rg_data.min_depth, rg_data.max_depth);
			traceRadiance(ray, payload);
			done = shade_material(payload, rg_data.min_depth,rg_data.max_depth, ray_org, ray_dir, color, throughput, random);
		}
	}

	auto res = optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x];
	auto col = (color + owl::vec3f(res));
	auto smp = res.w + frame_samples;
	optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x] = owl::vec4f(col, smp);
	col *= (1.0f / smp);
	optixLaunchParams.frame_buffer[dim.x * idx.y + idx.x] = col;
}
// レイタイプ: Radiance
// 最近傍シェーダ(サーフェス情報を取得)
OPTIX_CLOSEST_HIT_PROGRAM(radianceCH)() {
	auto& ch_data    = owl::getProgramData<HitgroupData>();
	auto& payload    = owl::getPRD<PayloadData>();
	auto pri_idx     = optixGetPrimitiveIndex();
	auto tri_idx     = ch_data.indices[pri_idx];
	auto v0          = ch_data.vertices[tri_idx.x];
	auto v1          = ch_data.vertices[tri_idx.y];
	auto v2          = ch_data.vertices[tri_idx.z];
	auto v01         = v1 - v0;
	auto v12         = v2 - v1;
	auto f_normal    = owl::normalize(owl::cross(v01, v12));
	auto bary        = optixGetTriangleBarycentrics();
	auto vt0         = ch_data.uvs[tri_idx.x];
	auto vt1         = ch_data.uvs[tri_idx.y];
	auto vt2         = ch_data.uvs[tri_idx.z];
	auto vt          = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
	payload.mat_idx  = ch_data.mat_idx+1;
	payload.texcoord = vt;
	payload.distance = optixGetRayTmax();
	payload.g_normal = f_normal;
	payload.s_normal = f_normal;
	// 実際の処理はCallableで実行する
}
OPTIX_CLOSEST_HIT_PROGRAM(radianceCHWithNormalShading)() {
	auto& ch_data           = owl::getProgramData<HitgroupData>();
	auto& payload           = owl::getPRD<PayloadData>();
	auto pri_idx            = optixGetPrimitiveIndex();
	auto tri_idx            = ch_data.indices[pri_idx];
	auto v0                 = ch_data.vertices[tri_idx.x];
	auto v1                 = ch_data.vertices[tri_idx.y];
	auto v2                 = ch_data.vertices[tri_idx.z];
	auto v01                = v1 - v0;
	auto v12                = v2 - v1;
	auto f_normal           = owl::normalize(owl::cross(v01, v12));
	auto bary               = optixGetTriangleBarycentrics();
	auto vn0                = ch_data.normals[tri_idx.x];
	auto vn1                = ch_data.normals[tri_idx.y];
	auto vn2                = ch_data.normals[tri_idx.z];
	auto vt0                = ch_data.uvs[tri_idx.x];
	auto vt1                = ch_data.uvs[tri_idx.y];
	auto vt2                = ch_data.uvs[tri_idx.z];
	auto vt                 = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
	auto v_normal           = owl::normalize((1.0f - (bary.x + bary.y)) * vn0 + bary.x * vn1 + bary.y * vn2);
	payload.mat_idx         = ch_data.mat_idx+1;
	payload.texcoord        = vt;
	payload.distance        = optixGetRayTmax();
	payload.g_normal        = f_normal;
	payload.s_normal        = v_normal;
	// 実際の処理はCallableで実行する
}
OPTIX_CLOSEST_HIT_PROGRAM(radianceCHWithNormalMap)() {
	auto& ch_data           = owl::getProgramData<HitgroupData>();
	auto& payload           = owl::getPRD<PayloadData>();
	auto pri_idx            = optixGetPrimitiveIndex();
	auto tri_idx            = ch_data.indices[pri_idx];
	auto v0                 = ch_data.vertices[tri_idx.x];
	auto v1                 = ch_data.vertices[tri_idx.y];
	auto v2                 = ch_data.vertices[tri_idx.z];
	auto v01                = v1 - v0;
	auto v12                = v2 - v1;
	auto f_normal           = owl::normalize(owl::cross(v01, v12));
	auto bary               = optixGetTriangleBarycentrics();
	auto vn0                = ch_data.normals[tri_idx.x];
	auto vn1                = ch_data.normals[tri_idx.y];
	auto vn2                = ch_data.normals[tri_idx.z];
	auto vt0                = ch_data.uvs[tri_idx.x];
	auto vt1                = ch_data.uvs[tri_idx.y];
	auto vt2                = ch_data.uvs[tri_idx.z];
	auto vt                 = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
	auto vn0_l              = owl::dot(vn0, vn0);
	auto vn1_l              = owl::dot(vn1, vn1);
	auto vn2_l              = owl::dot(vn2, vn2);
	auto v_normal           = owl::normalize((1.0f - (bary.x + bary.y)) * vn0 + bary.x * vn1 + bary.y * vn2);
	auto vtg0               = owl::vec3f(ch_data.tangents[tri_idx.x]);
	auto vtg1               = owl::vec3f(ch_data.tangents[tri_idx.y]);
	auto vtg2               = owl::vec3f(ch_data.tangents[tri_idx.z]);
	auto vbs0               = ch_data.tangents[tri_idx.x].w;
	auto vbs1               = ch_data.tangents[tri_idx.y].w;
	auto vbs2               = ch_data.tangents[tri_idx.z].w;
	auto vbn0               = vbs0 * owl::normalize(owl::cross(vn0, vtg0));
	auto vbn1               = vbs1 * owl::normalize(owl::cross(vn1, vtg1));
	auto vbn2               = vbs2 * owl::normalize(owl::cross(vn2, vtg2));
	auto v_binormal         = owl::normalize((1.0f - (bary.x + bary.y)) * vbn0 + bary.x * vbn1 + bary.y * vbn2);
	auto v_tangent          = owl::normalize(owl::cross(v_binormal, v_normal));
	auto tmp_bump           = tex2D<float4>(ch_data.texture_normal, vt.x, vt.y);
	payload.mat_idx         = ch_data.mat_idx+1;
	payload.texcoord        = vt;
	payload.distance        = optixGetRayTmax();
	payload.g_normal        = f_normal;
	payload.s_normal        = owl::normalize(tmp_bump.z * v_normal + (2.0f * tmp_bump.x - 1.0f) * v_tangent + (2.0f * tmp_bump.y - 1.0f) * v_binormal);
	// 実際の処理はCallableで実行する
}
// ミスシェーダ  (サーフェス情報を取得)
OPTIX_MISS_PROGRAM(radianceMS)() {
	auto& payload = owl::getPRD<PayloadData>();
	payload.mat_idx = 0;
	payload.texcoord = { 0.0f, 0.0f };
	payload.distance = 0.0f;
	payload.g_normal = { 0.0f,0.0f, 0.0f };
	payload.s_normal = { 0.0f,0.0f, 0.0f };
}
// レイタイプ: Occluded
// 最近傍シェーダ(可視情報を取得)
OPTIX_CLOSEST_HIT_PROGRAM(occludedCH)() {
	optixSetPayload_0(1);
}
// ミスシェーダ  (可視情報を取得)
OPTIX_MISS_PROGRAM(occludedMS)() {
	optixSetPayload_0(0);
}
// AnyHitシェーダ(αテストを起動）
OPTIX_ANY_HIT_PROGRAM(simpleAH)() {
	auto& ch_data = owl::getProgramData<HitgroupData>();
	if (optixIsTriangleHit()) {
		auto pri_idx = optixGetPrimitiveIndex();
		auto tri_idx = ch_data.indices[pri_idx];
		auto bary = optixGetTriangleBarycentrics();

		auto vt0 = ch_data.uvs[tri_idx.x];
		auto vt1 = ch_data.uvs[tri_idx.y];
		auto vt2 = ch_data.uvs[tri_idx.z];

		auto vt = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
		auto tmp_col = tex2D<float4>(ch_data.texture_alpha, vt.x, vt.y);
		if (tmp_col.w * tmp_col.x * tmp_col.y * tmp_col.z < 0.5f) {
			optixIgnoreIntersection();
		}
	}
}