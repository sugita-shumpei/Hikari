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
	owl::LCG<24> random = {};
	random.init(dim.x * idx.y + idx.x, optixLaunchParams.accum_sample);

	auto color = owl::vec3f(0.0f,0.0f,0.0f);
	for (int i = 0; i < frame_samples; ++i) {
		payload  = PayloadData();
		float px = ((float)idx.x + random() - 0.5f) / ((float)dim.x);
		float py = ((float)idx.y + random() - 0.5f) / ((float)dim.y);

		auto ray_dir = rg_data.camera.getRayDirection(px, py);
		owl::RayT<0, 1> ray(rg_data.camera.eye,
			owl::normalize(ray_dir),
			rg_data.min_depth, rg_data.max_depth
		);
		owl::trace(rg_data.world, ray, RAY_TYPE_COUNT, payload, RAY_TYPE_RADIANCE);
		color.x += (payload.s_normal.x + 1.0f) * 0.5f;
		color.y += (payload.s_normal.y + 1.0f) * 0.5f;
		color.z += (payload.s_normal.z + 1.0f) * 0.5f;
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
	payload.mat_idx       = 0;
	payload.texcoord      = {0.0f, 0.0f};
	payload.distance      =  0.0f;
	payload.g_normal      = { 0.0f,0.0f, 0.0f };
	payload.s_normal      = { 0.0f,0.0f, 0.0f };
}
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
	payload.mat_idx  = 0;
	payload.texcoord = vt;
	payload.distance = optixGetRayTmax();
	payload.g_normal = f_normal;
	payload.s_normal = f_normal;
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
	payload.mat_idx         = 0;
	payload.texcoord        = vt;
	payload.distance        = optixGetRayTmax();
	payload.g_normal        = f_normal;
	payload.s_normal        = v_normal;
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
	payload.mat_idx         = 0;
	payload.texcoord        = vt;
	payload.distance        = optixGetRayTmax();
	payload.g_normal        = f_normal;
	payload.s_normal        = owl::normalize(tmp_bump.z * v_normal + (2.0f * tmp_bump.x - 1.0f) * v_tangent + (2.0f * tmp_bump.y - 1.0f) * v_binormal);
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