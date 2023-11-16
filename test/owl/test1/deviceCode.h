#pragma once
#include <owl/common/math/vec.h>
#include <cuda.h>

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
};
struct MissProgData 
{

};
struct HitgroupData {

};
struct CallableData {
	owl::vec4f color;
};