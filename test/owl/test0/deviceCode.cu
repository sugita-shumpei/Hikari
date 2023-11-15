#include <optix_device.h>
#include <owl/owl_device.h>
#include <owl/common/owl-common.h>
#include "deviceCode.h"
 
extern "C" { __constant__ LaunchParams optixLaunchParams; }

OPTIX_RAYGEN_PROGRAM(simpleRG)() {
	const owl::vec2i idx = owl::getLaunchIndex();
	const owl::vec2i dim = owl::getLaunchDims();
	auto& rg_data = owl::getProgramData<RayGenData>();
	auto color = owl::vec3f(0.4f, 0.4f, 0.4f);
	owl::RayT<0,1> ray(owl::vec3f(
		2.0f*(((float)idx.x)/((float)dim.x))-1.0f,
		2.0f*(((float)idx.y)/((float)dim.y))-1.0f,
		0.0f),
		owl::vec3f(0.0f,0.0f,1.0f),
		0.01f,1000.0f
	);
	owl::trace(rg_data.world, ray, 1, color);
	rg_data.fb_data[dim.x * idx.y + idx.x] = owl::make_rgba(color);
}

OPTIX_MISS_PROGRAM(  simpleMS)() {
	auto& color = owl::getPRD<owl::vec3f>();
	color.x = 0.0f;
	color.y = 0.3f;
	color.z = 0.3f;
}

OPTIX_CLOSEST_HIT_PROGRAM(simpleCH)() {
	auto& color = owl::getPRD<owl::vec3f>();
	color.x = 1.0f;
	color.y = 0.0f;
	color.z = 0.0f;
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
