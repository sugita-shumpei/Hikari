#pragma once
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>
struct CameraData {
	owl::vec3f eye;
	owl::vec3f dir_u;
	owl::vec3f dir_v;
	owl::vec3f dir_w;
#if defined(__CUDACC__)
	__forceinline__ __device__ owl::vec3f getRayDirection(float u, float v)const {
		return dir_w + dir_u * (2.0f * u - 1.0f) + dir_v * (2.0f * v - 1.0f);
	}
#endif
};

struct LaunchParams
{
	OptixTraversableHandle world;
	owl::vec4f*            accum_buffer;
	int                    accum_sample;
};

struct RayGenData   
{
	OptixTraversableHandle world    ;
	uint32_t*              fb_data  ;
	owl::vec2i             fb_size  ;
	float                  min_depth;
	float                  max_depth;
	CameraData             camera   ;
};

struct MissProgData 
{
	cudaTextureObject_t texture_envlight;
};

struct HitgroupData {
	owl::vec3f *         vertices       ;
	owl::vec3f *         normals        ;
	owl::vec4f *         tangents       ;
	owl::vec2f *         uvs            ;
	owl::vec3f *         colors         ;
	owl::vec3ui*         indices        ;
	// phong 
	owl::vec3f           color_ambient  ;
	owl::vec3f           color_specular ;
	float                shininess      ;

	cudaTextureObject_t  texture_alpha   ; 
	cudaTextureObject_t  texture_ambient ;
	cudaTextureObject_t  texture_normal  ;   // (normal map)
	cudaTextureObject_t  texture_specular;
};

struct CallableData {
	owl::vec4f color;
};