#include <optix_device.h>
#include <owl/owl_device.h>
#include <owl/common/owl-common.h>
#include <owl/common/math/random.h>
#include "deviceCode.h"
 
extern "C" { __constant__ LaunchParams optixLaunchParams; }

struct PayloadData {
	owl::vec3f   color;
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

OPTIX_RAYGEN_PROGRAM(simpleRG)() {
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	auto& rg_data        = owl::getProgramData<RayGenData>();

	owl::LCG<24>           random;
	random.init(dim.x * idx.y + idx.x, optixLaunchParams.accum_sample);

	constexpr auto frame_samples = 1;

	auto color = owl::vec3f(0.0f,0.0f,0.0f);
	for (int i = 0; i < frame_samples; ++i) {
		float px      = ((float)idx.x + random() - 0.5f) / ((float)dim.x);
		float py      = ((float)idx.y + random() - 0.5f) / ((float)dim.y);
		auto  ray_dir = rg_data.camera.getRayDirection(px, py);

		owl::RayT<0, 1> ray(rg_data.camera.eye,
			owl::normalize(ray_dir),
			rg_data.min_depth, rg_data.max_depth
		);

		auto  payload = PayloadData();
		payload.color = owl::vec3f(0.0f, 0.0f, 0.0f);
		owl::trace(rg_data.world, ray, 1, payload);
		color += payload.color;
	}


	auto res = optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x];
	auto col = (color + owl::vec3f(res));
	auto smp = res.w + frame_samples;
	optixLaunchParams.accum_buffer[dim.x * idx.y + idx.x] = owl::vec4f(col, smp);
	col *= (1.0f / smp);
	rg_data.fb_data[dim.x * idx.y + idx.x] = owl::make_rgba(col);
}

OPTIX_MISS_PROGRAM(  simpleMS)() {
	auto& payload = owl::getPRD<PayloadData>();
	auto& ms_data = owl::getProgramData<MissProgData>();
	auto  ray_dir = owl::normalize(owl::vec3f(optixGetWorldRayDirection()));
	float phi     = atan2f(ray_dir.z, ray_dir.x);
	float theta   = asinf (ray_dir.y)  ;
	float x       = (phi   / M_PI + 1.0f)*0.5f;
	float y       = (theta / M_PI + 0.5f);

	// x y z
	//x=(ray_dir.x + 1.0f)*0.5f
	//y=(ray_dir.y + 1.0f)*0.5f
	auto tmp_col = tex2D<float4>(ms_data.texture_envlight ,x, y);
	payload.color.x = tmp_col.x;
	payload.color.y = tmp_col.y;
	payload.color.z = tmp_col.z;
}

OPTIX_ANY_HIT_PROGRAM(simpleAH)() {
	auto& ch_data = owl::getProgramData<HitgroupData>();
	if (optixIsTriangleHit()) {
		auto pri_idx = optixGetPrimitiveIndex();
		auto tri_idx = ch_data.indices[pri_idx];
		auto bary    = optixGetTriangleBarycentrics();

		auto vt0 = ch_data.uvs[tri_idx.x];
		auto vt1 = ch_data.uvs[tri_idx.y];
		auto vt2 = ch_data.uvs[tri_idx.z];

		auto vt = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);
		auto tmp_col = tex2D<float4>(ch_data.texture_alpha, vt.x, vt.y);
		if (tmp_col.w * tmp_col.x * tmp_col.y* tmp_col.z < 0.5f) {
			optixIgnoreIntersection();
		}
	}
}

OPTIX_CLOSEST_HIT_PROGRAM(simpleCH)() {
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
	auto vn0 = ch_data.normals[tri_idx.x];
	auto vn1 = ch_data.normals[tri_idx.y];
	auto vn2 = ch_data.normals[tri_idx.z];

	auto vt0 = ch_data.uvs[tri_idx.x];
	auto vt1 = ch_data.uvs[tri_idx.y];
	auto vt2 = ch_data.uvs[tri_idx.z];

	auto vt = normalize_uv((1.0f - (bary.x + bary.y)) * vt0 + bary.x * vt1 + bary.y * vt2);

	auto vn0_l = owl::dot(vn0, vn0);
	auto vn1_l = owl::dot(vn1, vn1);
	auto vn2_l = owl::dot(vn2, vn2);
	if (vn0_l < 0.01f) { vn0 = f_normal; }
	if (vn1_l < 0.01f) { vn1 = f_normal; }
	if (vn2_l < 0.01f) { vn2 = f_normal; }
	// 
	auto v_normal = owl::normalize((1.0f - (bary.x + bary.y)) * vn0 + bary.x * vn1 + bary.y * vn2);
	//payload.color.x = (v_normal.x + 1.0f) * 0.5f;
	//payload.color.y = (v_normal.y + 1.0f) * 0.5f;
	//payload.color.z = (v_normal.z + 1.0f) * 0.5f;
	auto vtg0 = owl::vec3f(ch_data.tangents[tri_idx.x]);
	auto vtg1 = owl::vec3f(ch_data.tangents[tri_idx.y]);
	auto vtg2 = owl::vec3f(ch_data.tangents[tri_idx.z]);

	auto vbs0 = ch_data.tangents[tri_idx.x].w;
	auto vbs1 = ch_data.tangents[tri_idx.y].w;
	auto vbs2 = ch_data.tangents[tri_idx.z].w;
	auto vbn0 =  vbs0 * owl::normalize(owl::cross(vn0, vtg0));
	auto vbn1 =  vbs1 * owl::normalize(owl::cross(vn1, vtg1));
	auto vbn2 =  vbs2 * owl::normalize(owl::cross(vn2, vtg2));
	auto v_binormal   = owl::normalize((1.0f - (bary.x + bary.y)) * vbn0 + bary.x * vbn1 + bary.y * vbn2);
	//payload.color.x = (v_binormal.x + 1.0f) * 0.5f;
	//payload.color.y = (v_binormal.y + 1.0f) * 0.5f;
	//payload.color.z = (v_binormal.z + 1.0f) * 0.5f;

	auto v_tangent = owl::normalize(owl::cross(v_binormal, v_normal));

	auto tmp_bump = tex2D<float4>(ch_data.texture_normal, vt.x, vt.y);
	// shading�@��(�����܂ŕ`��p)
	auto fin_normal = owl::normalize(tmp_bump.z * v_normal + (2.0f * tmp_bump.x - 1.0f) * v_tangent + (2.0f * tmp_bump.y - 1.0f) * v_binormal);

	//payload.color.x = (fin_normal.x + 1.0f) * 0.5f;
	//payload.color.y = (fin_normal.y + 1.0f) * 0.5f;
	//payload.color.z = (fin_normal.z + 1.0f) * 0.5f;

	auto ambient_col = ch_data.color_ambient *owl::vec3f(tex2D<float4>(ch_data.texture_ambient, vt.x, vt.y));

	//color = ch_data.colors[pri_idx];
	payload.color.x = ambient_col.x;
	payload.color.y = ambient_col.y;
	payload.color.z = ambient_col.z;
}

OPTIX_DIRECT_CALLABLE_PROGRAM(simpleDC1)(owl::vec4f& c) {
	auto& callable_data  = owl::getProgramData<CallableData>();
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	c = callable_data.color;
}

OPTIX_DIRECT_CALLABLE_PROGRAM(simpleDC2)(owl::vec4f& c) {
	auto& callable_data  = owl::getProgramData<CallableData>();
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	c = callable_data.color;
}
