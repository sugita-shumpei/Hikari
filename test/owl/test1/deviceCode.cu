#include <optix_device.h>
#include <owl/owl_device.h>
#include <owl/common/owl-common.h>
#include <owl/common/math/random.h>
#include <owl/common/math/vec.h>
#include "deviceCode.h"
extern "C" { __constant__ LaunchParams optixLaunchParams; }
struct PayloadData {
	owl::LCG<24> random     ;
	owl::vec3f ray_origin   ;
	owl::vec3f ray_direction;
	owl::vec3f attenuation  ;
	owl::vec3f color;
};
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

OPTIX_RAYGEN_PROGRAM(simpleRG)() {
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	auto& rg_data        = owl::getProgramData<RayGenData>();

	constexpr auto frame_samples = 2;
	constexpr auto trace_depth   = 8;

	auto payload = PayloadData();
	payload.random = {};
	payload.random.init(dim.x * idx.y + idx.x, optixLaunchParams.accum_sample);

	auto color = owl::vec3f(0.0f,0.0f,0.0f);
	for (int i = 0; i < frame_samples; ++i) {
		payload.color       = owl::vec3f(0.0f, 0.0f, 0.0f);
		payload.attenuation = owl::vec3f(1.0f, 1.0f, 1.0f);

		float px = ((float)idx.x + payload.random() - 0.5f) / ((float)dim.x);
		float py = ((float)idx.y + payload.random() - 0.5f) / ((float)dim.y);

		auto ray_dir = rg_data.camera.getRayDirection(px, py);
		owl::RayT<0, 1> ray(rg_data.camera.eye,
			owl::normalize(ray_dir),
			rg_data.min_depth, rg_data.max_depth
		);

		for (int j = 0; j < trace_depth; ++j) {
			payload.color = owl::vec3f(0.0f, 0.0f, 0.0f);
			owl::trace(rg_data.world, ray, RAY_TYPE_COUNT, payload, RAY_TYPE_RADIANCE);
			ray.origin    = payload.ray_origin;
			ray.direction = payload.ray_direction;
			color        += payload.color;
		}
	}

	auto res = optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x];
	auto col = (color + owl::vec3f(res));
	auto smp = res.w + frame_samples;
	optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x] = owl::vec4f(col, smp);
	col *= (1.0f / smp);
	optixLaunchParams.frame_buffer[dim.x * idx.y + idx.x] = col;
}

OPTIX_MISS_PROGRAM(occludedMS)() {
	optixSetPayload_0(0);
}

OPTIX_MISS_PROGRAM(radianceMS)() {
	auto& payload         = owl::getPRD<PayloadData>();
	auto& ms_data         = owl::getProgramData<MissProgData>();
	auto  emission        = optixLaunchParams.light_intensity*owl::vec3f(sample_sphere_map(optixLaunchParams.light_envmap, optixGetWorldRayDirection()));
	payload.color.x       = payload.attenuation.x * emission.x;
	payload.color.y       = payload.attenuation.y * emission.y;
	payload.color.z       = payload.attenuation.z * emission.z;
	payload.attenuation.x = 0.0f;
	payload.attenuation.y = 0.0f;
	payload.attenuation.z = 0.0f;
}
OPTIX_CLOSEST_HIT_PROGRAM(radianceCH)() {
	auto& ch_data = owl::getProgramData<HitgroupData>();
	auto& payload = owl::getPRD<PayloadData>();

	auto pri_idx = optixGetPrimitiveIndex();
	auto tri_idx = ch_data.indices[pri_idx];
	auto v0 = ch_data.vertices[tri_idx.x];
	auto v1 = ch_data.vertices[tri_idx.y];
	auto v2 = ch_data.vertices[tri_idx.z];
	auto v01 = v1 - v0;
	auto v12 = v2 - v1;
	auto f_normal = owl::normalize(owl::cross(v01, v12));

	auto bary = optixGetTriangleBarycentrics();
	auto vn0  = ch_data.normals[tri_idx.x];
	auto vn1  = ch_data.normals[tri_idx.y];
	auto vn2  = ch_data.normals[tri_idx.z];

	auto vt0          = ch_data.uvs[tri_idx.x];
	auto vt1          = ch_data.uvs[tri_idx.y];
	auto vt2          = ch_data.uvs[tri_idx.z];

	auto vt           = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);

	auto vn0_l        = owl::dot(vn0, vn0);
	auto vn1_l        = owl::dot(vn1, vn1);
	auto vn2_l        = owl::dot(vn2, vn2);

	if (vn0_l < 0.01f) { vn0 = f_normal; }
	if (vn1_l < 0.01f) { vn1 = f_normal; }
	if (vn2_l < 0.01f) { vn2 = f_normal; }
	// 
	auto v_normal     = owl::normalize((1.0f - (bary.x + bary.y)) * vn0 + bary.x * vn1 + bary.y * vn2);
	//payload.color.x = (v_normal.x + 1.0f) * 0.5f;
	//payload.color.y = (v_normal.y + 1.0f) * 0.5f;
	//payload.color.z = (v_normal.z + 1.0f) * 0.5f;
	auto vtg0         = owl::vec3f(ch_data.tangents[tri_idx.x]);
	auto vtg1         = owl::vec3f(ch_data.tangents[tri_idx.y]);
	auto vtg2         = owl::vec3f(ch_data.tangents[tri_idx.z]);

	auto vbs0         = ch_data.tangents[tri_idx.x].w;
	auto vbs1         = ch_data.tangents[tri_idx.y].w;
	auto vbs2         = ch_data.tangents[tri_idx.z].w;
	
	auto vbn0         = vbs0 * owl::normalize(owl::cross(vn0, vtg0));
	auto vbn1         = vbs1 * owl::normalize(owl::cross(vn1, vtg1));
	auto vbn2         = vbs2 * owl::normalize(owl::cross(vn2, vtg2));
	auto v_binormal   = owl::normalize((1.0f - (bary.x + bary.y)) * vbn0 + bary.x * vbn1 + bary.y * vbn2);

	auto v_tangent    = owl::normalize(owl::cross(v_binormal, v_normal));

	auto tmp_bump     = tex2D<float4>(ch_data.texture_normal, vt.x, vt.y);
	//// shading�@��(�����܂ŕ`��p)
	auto fin_normal   = owl::normalize(tmp_bump.z * v_normal + (2.0f * tmp_bump.x - 1.0f) * v_tangent + (2.0f * tmp_bump.y - 1.0f) * v_binormal);
	//payload.color.x = (f_normal.x + 1.0f) * 0.5f;
	//payload.color.y = (f_normal.y + 1.0f) * 0.5f;
	//payload.color.z = (f_normal.z + 1.0f) * 0.5f;
	auto ambient_col  = ch_data.color_ambient *owl::vec3f(tex2D<float4>(ch_data.texture_ambient, vt.x, vt.y));
	auto specular_col = ch_data.color_specular*owl::vec3f(tex2D<float4>(ch_data.texture_specular, vt.x, vt.y));
	auto shininess    = ch_data.shininess;

	auto old_direction  = owl::vec3f(optixGetWorldRayDirection());
	auto normal         = fin_normal;
	auto ref_direction  = owl::normalize(old_direction - 2.0f * owl::dot(normal, old_direction) * normal);
	auto new_direction  = owl::vec3f(0.0f, 0.0f, 0.0f);
	{
		auto  origin    = owl::vec3f(optixGetWorldRayOrigin()) + optixGetRayTmax() * old_direction;
		float cos_theta = fmaxf(1.0f - payload.random(), 0.0f);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
		float phi       = 2.0f * M_PI * payload.random();
		float cos_phi   = cosf(phi);
		float sin_phi   = sinf(phi);

		Onb onb(normal);
		new_direction          = owl::normalize(onb.local({sin_theta*cos_phi,sin_theta*sin_phi,cos_theta}));
		float geometry_cosine  = cos_theta;

		payload.ray_origin    = origin + 0.001f * normal;
		payload.ray_direction = new_direction;
		payload.color         = fmaxf(geometry_cosine,0.0f) * payload.attenuation * ch_data.color_emission;

		if (optixLaunchParams.light_parallel.active) {
			// 並行光源へとレイを照射し遮蔽をとる
			unsigned int occluded = 0;
			optixTrace(
				optixLaunchParams.world,
				{ payload.ray_origin.x,payload.ray_origin.y,payload.ray_origin.z },
				{ optixLaunchParams.light_parallel.direction.x, optixLaunchParams.light_parallel.direction.y, optixLaunchParams.light_parallel.direction.z },
				0.0f, 1e10f, 0.0f, 255u,
				OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				RAY_TYPE_OCCLUDED,
				RAY_TYPE_COUNT,
				RAY_TYPE_OCCLUDED,
				occluded);
			// 遮蔽がなければ寄与をとる
			if (!occluded) {
				auto light_cosine = fmaxf(owl::dot(ref_direction, optixLaunchParams.light_parallel.direction), 0.0f);
				payload.color    += payload.attenuation * optixLaunchParams.light_parallel.color * eval_bsdf_phong(ambient_col, specular_col, shininess, light_cosine) * fmaxf(owl::dot(normal,optixLaunchParams.light_parallel.direction), 0.0f);
			}
		}

		if (geometry_cosine > 0.0f) {
			auto light_cosine = fmaxf(owl::dot(ref_direction, new_direction), 0.0f);
			payload.attenuation *= eval_bsdf_phong(ambient_col,specular_col,shininess, light_cosine) * ((float)M_PI);
		}
		else {
			payload.attenuation  = {};
		}
	}

}
OPTIX_CLOSEST_HIT_PROGRAM(occludedCH)() {
	optixSetPayload_0(1);
}

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