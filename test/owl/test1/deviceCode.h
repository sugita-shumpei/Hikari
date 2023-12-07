#pragma once
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

enum LightType {
	LIGHT_TYPE_ENVMAP,
	LIGHT_TYPE_DIRECTIONAL,
	LIGHT_TYPE_COUNT,
};

enum CallableType {
	CALLABLE_TYPE_SAMPLE_LIGHT             = 0 ,
	CALLABLE_TYPE_SAMPLE_LIGHT_ENVMAP      = CALLABLE_TYPE_SAMPLE_LIGHT + LIGHT_TYPE_ENVMAP     ,
	CALLABLE_TYPE_SAMPLE_LIGHT_DIRECTIONAL = CALLABLE_TYPE_SAMPLE_LIGHT + LIGHT_TYPE_DIRECTIONAL,
	CALLABLE_TYPE_COUNT
};

enum RayType {
	RAY_TYPE_RADIANCE,
	RAY_TYPE_OCCLUDED,
	RAY_TYPE_COUNT
};

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

struct ParallelLight {
	owl::vec3f             color    ;
	unsigned int           active   ;
	owl::vec3f             direction;
	unsigned int           dummy    ;
};

struct LaunchParams
{ 
	OptixTraversableHandle world             ;
	cudaTextureObject_t    light_envmap      ;
	owl::vec3f*            frame_buffer      ;
	owl::vec4f*            accum_buffer      ;
	owl::vec2i             frame_size        ;
	int                    accum_sample      ;
	float                  light_intensity   ;
	ParallelLight          light_parallel    ;
};

struct RayGenData   
{
	OptixTraversableHandle world    ;
	CameraData             camera   ;
	float                  min_depth;
	float                  max_depth;
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
	owl::vec3f           color_emission ;
	float                shininess      ;

	cudaTextureObject_t  texture_alpha   ; 
	cudaTextureObject_t  texture_ambient ;
	cudaTextureObject_t  texture_normal  ;   // (normal map)
	cudaTextureObject_t  texture_specular;
};

struct Onb {
#if defined(__CUDACC__)
	__device__ 
#endif
	Onb(const owl::vec3f& w_) noexcept: w { w_ }{
		if (fabsf(w.x) < 0.5f) {
			u = owl::cross(w, owl::vec3f(1.0f, 0.0f, 0.0f));
		}
		else if (fabsf(w.y) < 0.5f) {
			u = owl::cross(w, owl::vec3f(0.0f, 1.0f, 0.0f));
		}
		else {
			u = owl::cross(w, owl::vec3f(0.0f, 0.0f, 1.0f));
		}
		v = owl::cross(w, u);
	}

#if defined(__CUDACC__)
	__forceinline__ __device__ 
#endif
	owl::vec3f local(const owl::vec3f& direction) const {
		return direction.x * u + direction.y * v + direction.z * w;
	}


	owl::vec3f u;
	owl::vec3f v;
	owl::vec3f w;
};
