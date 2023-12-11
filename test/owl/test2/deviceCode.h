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

enum   MaterialType {
	MATERIAL_TYPE_NONE                         ,// シェーディングを行わない
	MATERIAL_TYPE_LIGHT                        ,// 光源
	MATERIAL_TYPE_DIFFUSE_LAMBERT              ,// ランバート拡散モデル
	MATERIAL_TYPE_SPECULAR_ROUGH_PHONG         ,// フォン反射モデル
	MATERIAL_TYPE_SPECULAR_DELTA               ,// 完全鏡面
	MATERIAL_TYPE_DIELECTRIC_DELTA             ,// 完全誘電体
	MATERIAL_TYPE_LEGACY_PHONG_COMPOSITE       ,// 古典的Phong+LambertBRDF
	MATERIAL_TYPE_THIN_DIELECTRIC_DELTA        ,// 薄い完全誘電体
	// 物理ベースレンダリング
	MATERIAL_TYPE_BLEND                        ,// ２つのマテリアルを一定割合で混ぜる
	MATERIAL_TYPE_SPECULAR_ROUGH_BECKMAN       ,// Beckman分布に基づく荒い反射
	MATERIAL_TYPE_SPECULAR_ROUGH_GGX           ,// GGX分布に基づく荒い反射
	MATERIAL_TYPE_DIELECTRIC_ROUGH_BECKMAN     ,// Beckman分布に基づく荒い誘電体
	MATERIAL_TYPE_DIELECTRIC_ROUGH_GGX         ,// GGX分布に基づく荒い誘電体
	MATERIAL_TYPE_THIN_DIELECTRIC_ROUGH_BECKMAN,// Beckman分布に基づく荒い薄い誘電体
	MATERIAL_TYPE_THIN_DIELECTRIC_ROUGH_GGX    ,// GGX分布に基づく荒い薄い誘電体
};
struct MaterialParams {
	unsigned int          material_type ;
	float                 f32_values[11];//32bit*12
	unsigned short        u16_values[16];//32bit* 8
};

//  Material Parameter
struct MaterialParamsLight {
#if defined(__CUDACC__)
	__device__ void set(const MaterialParams& params) {
		color.x = params.f32_values[0];
		color.y = params.f32_values[1];
		color.z = params.f32_values[2];
		texid   = params.u16_values[0];
	}
#else
	void get(MaterialParams& params)const {
		params.f32_values[0] = color.x;
		params.f32_values[1] = color.y;
		params.f32_values[2] = color.z;
		params.u16_values[0] = texid;
	}
#endif
	owl::vec3f            color;
	unsigned short        texid;
};
struct MaterialParamsDiffuse {
	owl::vec3f            color;
	unsigned short        texid;
#if defined(__CUDACC__)
	__device__ void set(const MaterialParams& params) {
		color.x = params.f32_values[0];
		color.y = params.f32_values[1];
		color.z = params.f32_values[2];
		texid = params.u16_values[0];
	}
#else
	void get(MaterialParams& params)const {
		params.f32_values[0] = color.x;
		params.f32_values[1] = color.y;
		params.f32_values[2] = color.z;
		params.u16_values[0] = texid;
	}
#endif
};
struct MaterialParamsSpecular {
	owl::vec3f            color;
	float                 factor1;
	float                 factor2;
	unsigned short        texid_color;
	unsigned short        texid_factor1;
	unsigned short        texid_factor2;
#if defined(__CUDACC__)
	__device__ void set(const MaterialParams& params) {
		color.x       = params.f32_values[0];
		color.y       = params.f32_values[1];
		color.z       = params.f32_values[2];
		factor1       = params.f32_values[3];
		factor2       = params.f32_values[4];
		texid_color   = params.u16_values[0];
		texid_factor1 = params.u16_values[1];
		texid_factor2 = params.u16_values[2];
	}
#else
	void get(MaterialParams& params)const {
		params.f32_values[0] = color.x;
		params.f32_values[1] = color.y;
		params.f32_values[2] = color.z;
		params.f32_values[3] = factor1;
		params.f32_values[4] = factor2;
		params.u16_values[0] = texid_color;
		params.u16_values[1] = texid_factor1;
		params.u16_values[2] = texid_factor2;
	}
#endif
};
struct MaterialParamsDielectric {
	float                 ior;
#if defined(__CUDACC__)
	__device__ void set(const MaterialParams& params) {
		ior = params.f32_values[0];
	}
#else
	void get(MaterialParams& params)const {
		params.f32_values[0] = ior;
	}
#endif
};
struct MaterialParamsLagacyPhongComposite {
#if defined(__CUDACC__)
	__device__ void set(const MaterialParams& params) {
		color_ambient.x = params.f32_values[0];
		color_ambient.y = params.f32_values[1];
		color_ambient.z = params.f32_values[2];
		color_specular.x = params.f32_values[3];
		color_specular.y = params.f32_values[4];
		color_specular.z = params.f32_values[5];
		shininess        = params.f32_values[6];
		texid_color_ambient   = params.u16_values[0];
		texid_color_specular  = params.u16_values[1];
		texid_color_shininess = params.u16_values[2];
	}
#else
	void get(MaterialParams& params)const {
		params.f32_values[0] = color_ambient.x;
		params.f32_values[1] = color_ambient.y;
		params.f32_values[2] = color_ambient.z;
		params.f32_values[3] = color_specular.x;
		params.f32_values[4] = color_specular.y;
		params.f32_values[5] = color_specular.z;
		params.f32_values[6] = shininess;
		params.u16_values[0] = texid_color_ambient;
		params.u16_values[1] = texid_color_specular;
		params.u16_values[2] = texid_color_shininess;
	}
#endif
	owl::vec3f            color_ambient;
	owl::vec3f            color_specular;
	float                 shininess;
	unsigned short        texid_color_ambient;
	unsigned short        texid_color_specular;
	unsigned short        texid_color_shininess;
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
	MaterialParams*        material_buffer   ;
	CUtexObject*           texture_buffer    ;
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
	owl::vec3f *         vertices      ;
	owl::vec3f *         normals       ;
	owl::vec4f *         tangents      ;
	owl::vec2f *         uvs           ;
	owl::vec3f *         colors        ;
	owl::vec3ui*         indices       ;
	cudaTextureObject_t  texture_alpha ; 
	cudaTextureObject_t  texture_normal;
	unsigned short       mat_idx       ;
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
