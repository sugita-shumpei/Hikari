#pragma once
#include <owl/common/math/vec.h>
#include <cuda.h>
struct CameraData {
	owl::vec3f eye;
	owl::vec3f dir_u;
	owl::vec3f dir_v;
	owl::vec3f dir_w;
#if defined(__CUDACC__)
	__forceinline__ __device__ owl::vec3f getRayDirection(float u, float v)const {
		return owl::normalize(dir_w + dir_u * (2.0f * u - 1.0f) + dir_v * (2.0f * v - 1.0f));
	}
#endif
};
struct LaunchParams
{
	OptixTraversableHandle world;
	owl::vec4f*     accum_buffer;
	int             accum_sample;
};
struct RayGenData   
{
	OptixTraversableHandle world;
	uint32_t*            fb_data;
	owl::vec2i           fb_size;
	CameraData            camera;
};
struct MissProgData 
{

};
struct HitgroupData {

};
struct CallableData {
	owl::vec4f color;
};