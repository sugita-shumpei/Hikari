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
	auto  payload        = PayloadData();
	payload.color        = owl::vec3f(0.4f, 0.4f, 0.4f);
	float px = ((float)idx.x + 0.5f) / ((float)dim.x);
	float py = ((float)idx.y + 0.5f) / ((float)dim.y);
	float cx = 2.0f * px - 1.0f;
	float cy = 2.0f * py - 1.0f;
	auto  ray_dir = rg_data.camera.getRayDirection(cx, cy);

	owl::RayT<0,1> ray(rg_data.camera.eye,
		owl::normalize(ray_dir),
		rg_data.min_depth,rg_data.max_depth
	);
	owl::trace(rg_data.world, ray, 1, payload);
	rg_data.fb_data[dim.x * idx.y + idx.x] = owl::make_rgba(payload.color);
}

OPTIX_MISS_PROGRAM(  simpleMS)() {
	auto& payload = owl::getPRD<PayloadData>();
	payload.color.x = 0.0f;
	payload.color.y = 0.3f;
	payload.color.z = 0.3f;
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

	auto pri_idx  = optixGetPrimitiveIndex();
	auto tri_idx  = ch_data.indices[pri_idx];
	auto v0       = ch_data.vertices[tri_idx.x];
	auto v1       = ch_data.vertices[tri_idx.y];
	auto v2       = ch_data.vertices[tri_idx.z];
	auto v01      = v1 - v0;
	auto v12      = v2 - v1;
	auto f_normal = owl::normalize(owl::cross(v01, v12));

	auto bary     = optixGetTriangleBarycentrics();
	auto vn0      = ch_data.normals[tri_idx.x];
	auto vn1      = ch_data.normals[tri_idx.y];
	auto vn2      = ch_data.normals[tri_idx.z];

	auto vt0      = ch_data.uvs[tri_idx.x];
	auto vt1      = ch_data.uvs[tri_idx.y];
	auto vt2      = ch_data.uvs[tri_idx.z];

	auto vt       = normalize_uv((1.0f - (bary.x + bary.y))* vt0 + bary.x * vt1 + bary.y * vt2);

	auto vn0_l = owl::dot(vn0, vn0);
	auto vn1_l = owl::dot(vn1, vn1);
	auto vn2_l = owl::dot(vn2, vn2);
	if (vn0_l < 0.01f) { vn0 = f_normal; }
	if (vn1_l < 0.01f) { vn1 = f_normal; }
	if (vn2_l < 0.01f) { vn2 = f_normal; }

	auto v_normal = owl::normalize((1.0f - (bary.x + bary.y)) * vn0 + bary.x * vn1 + bary.y * vn2);
	//color.x = (f_normal.x + 1.0f) * 0.5f;
	//color.y = (f_normal.y + 1.0f) * 0.5f;
	//color.z = (f_normal.z + 1.0f) * 0.5f;

	//color.x = vt.x;
	//color.y = vt.y;
	//color.z = 1.0f-(vt.x+vt.y)*0.5f;

	auto tmp_col = tex2D<float4>(ch_data.texture_ambient, vt.x, vt.y);

	//color = ch_data.colors[pri_idx];
	payload.color.x = tmp_col.x;
	payload.color.y = tmp_col.y;
	payload.color.z = tmp_col.z;
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
