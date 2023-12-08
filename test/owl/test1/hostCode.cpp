#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <filesystem>
#include <stb_image.h>
#include <stb_image_write.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>
#include "hostCode.h"
#include "deviceCode.h"

extern "C" char* deviceCode_ptx[];

int main() {

	hikari::test::owl::testlib::ObjModel model;
	// model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\CornellBox\CornellBox-Original.obj)");
    model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Sponza\sponza.obj)");
	// model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Bistro\Exterior\exterior.obj)");/*
	// auto envlit_filename = std::string(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Textures\evening_road_01_puresky_8k.hdr)");
	auto envlit_filename = std::string(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Textures\kloofendal_43d_clear_puresky_4k.hdr)");
    //auto envlit_filename = std::string("");
	auto center          = model.getBBox().getCenter();
	auto range           = model.getBBox().getRange ();
	hikari::test::owl::testlib::PinholeCamera camera;
	camera.origin.x      = center.x;
	camera.origin.y      = center.y;
	camera.origin.z      =  5.0f;
	camera.direction.z   = -1.0f;
	camera.speed.x       = range.x * 0.01f / 2.0f;
	camera.speed.y       = range.y * 0.01f / 2.0f;
	camera.speed.z       = range.z * 0.01f / 2.0f;

	auto bbox_len = 2.0f*std::sqrtf(range.x * range.x + range.y * range.y + range.z * range.z);
	auto context  = owlContextCreate();
	owlContextSetRayTypeCount(context, RAY_TYPE_COUNT);
	auto textures = std::vector<OWLTexture>();

	owl::vec3f     envlit_sun_dir     = { 0.0f,1.0f,0.0f };
	float          envlit_sun_k       = 1.0f;

	constexpr auto bump_level         = 5.0f;
	constexpr auto texture_idx_envlit = 0;
	constexpr auto texture_idx_black  = 1;
	constexpr auto texture_idx_white  = 2;
	constexpr auto texture_idx_blue   = 3;
	constexpr auto texture_idx_offset = 3;
	{
		// env lit  
		if (!envlit_filename.empty()){
			int w, h, c;
			auto pixels     = stbi_loadf(envlit_filename.data(), &w, &h, &c, 0);
			assert(pixels);
			auto pixel_data = std::vector<owl::vec4f>();
			pixel_data.resize(w * h, { 0.0f,0.0f,0.0f,0.0f });
			{
				owl::vec2f env_light_sun_uv = { 0.0f,0.0f };
				float max_luminance         = 0.0f;
				for (size_t i = 0; i < h; ++i) {
					for (size_t j = 0; j < w; ++j) {
						if (c == 1) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[1 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[1 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].z = pixels[1 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].w = 0.0f;
						}
						if (c == 2) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[2 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[2 * (i * w + j) + 1];
							pixel_data[(h - 1u - i) * w + j].z = 0;
							pixel_data[(h - 1u - i) * w + j].w = 0.0f;
						}
						if (c == 3) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[3 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[3 * (i * w + j) + 1];
							pixel_data[(h - 1u - i) * w + j].z = pixels[3 * (i * w + j) + 2];
							pixel_data[(h - 1u - i) * w + j].w = 0.0f;
						}
						if (c == 4) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[4 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[4 * (i * w + j) + 1];
							pixel_data[(h - 1u - i) * w + j].z = pixels[4 * (i * w + j) + 2];
							pixel_data[(h - 1u - i) * w + j].w = 0.0f;
						}
						
						float luminance    = 0.1769f * pixel_data[(h - 1u - i) * w + j].x + 0.8124f * pixel_data[(h - 1u - i) * w + j].y + 0.0107f * pixel_data[(h - 1u - i) * w + j].z;
						
						if (max_luminance < luminance) {
							max_luminance = luminance;
							env_light_sun_uv.x = static_cast<float>(j) / w;
							env_light_sun_uv.y = 1.0f-static_cast<float>(i) / h;
						}
					}
				}
				// 並行光源のサンプリング
				float phi           = M_PI * (2.0f * env_light_sun_uv.x-1.0f);
				float cos_theta     = 2.0f * env_light_sun_uv.y -1.0f;
				float sin_theta     = sqrtf(1.0f - cos_theta * cos_theta);
				envlit_sun_dir.x    = sin_theta * cosf(phi);
				envlit_sun_dir.y    = cos_theta;
				envlit_sun_dir.z    = sin_theta * sinf(phi);
			}
			auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, w, h, pixel_data.data(), OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
			textures.push_back(tex);

			stbi_image_free(pixels);
		}
		else {
			auto pixel = owl::vec4f(0, 0, 0, 0);
			auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 1, 1, &pixel);
			textures.push_back(tex);
		}
		// black
		{
			auto pixel = owl::vec4uc(0, 0, 0, 0);
			auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, 1, 1, &pixel);
			textures.push_back(tex);
		}
		// white
		{
			auto pixel = owl::vec4uc(255, 255, 255, 255);
			auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, 1, 1, &pixel);
			textures.push_back(tex);
		}
		// blue
		{
			auto pixel = owl::vec4uc(127u, 127u, 255, 255);
			auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, 1, 1, &pixel);
			textures.push_back(tex);
		}
		// model
		{
			
			for (auto& texture : model.getTextures()) {
				
				if (texture.filename == "") { continue; }
				bool is_bump_map = texture.filename.find("bump") != std::string::npos;
				auto filepath     = std::filesystem::canonical(std::filesystem::path(model.getFilename()).parent_path() / texture.filename);
				auto filepath_str = filepath.string();
				int w, h, c;
				auto pixels      = stbi_load(filepath_str.data(), &w, &h, &c, 0);
				assert(pixels);
				auto pixel_data = std::vector<owl::vec4uc>();
				pixel_data.resize(w * h, {0,0,0,0});
				if  (is_bump_map){
					// 3x3 sobel filterをかける
					for (size_t i = 0; i < h; ++i) {
						for (size_t j = 0; j < w; ++j) {
							if (i == 0 || i == h - 1) {
								pixel_data[(h - 1u - i) * w + j].x = 127u;
								pixel_data[(h - 1u - i) * w + j].y = 127u;
								pixel_data[(h - 1u - i) * w + j].z = 255u;
								pixel_data[(h - 1u - i) * w + j].w = 255u;
								continue; 
							}
							if (j == 0 || j == w - 1) {
								pixel_data[(h - 1u - i) * w + j].x = 127u;
								pixel_data[(h - 1u - i) * w + j].y = 127u;
								pixel_data[(h - 1u - i) * w + j].z = 255u;
								pixel_data[(h - 1u - i) * w + j].w = 255u;
								continue;
							}

							auto m_L0 = pixels[c * (i * w + j - 1) + 0];
							auto m_R0 = pixels[c * (i * w + j + 1) + 0];

							auto m_LL = pixels[c * (w * (i-1) + j - 1) + 0];
							auto m_0L = pixels[c * (w * (i-1) + j + 0) + 0];
							auto m_RL = pixels[c * (w * (i-1) + j + 1) + 0];
							
							auto m_LR = pixels[c * (w * (i+1) + j - 1) + 0];
							auto m_0R = pixels[c * (w * (i+1) + j + 0) + 0];
							auto m_RR = pixels[c * (w * (i+1) + j + 1) + 0];
							// 0 1020 2040
							auto du = (2u * m_R0 + m_RR + m_RL) + 255u * 4u - (2u * m_L0 + m_LR + m_LL);
							auto dv = (2u * m_0R + m_RR + m_LR) + 255u * 4u - (2u * m_0L + m_RL + m_LL);

							float nx = std::fminf(std::fmaxf((static_cast<float>(du) / static_cast<float>(1020) - 1.0f) * bump_level, -1.0f), 1.0f);
							float ny = std::fminf(std::fmaxf((static_cast<float>(dv) / static_cast<float>(1020) - 1.0f) * bump_level, -1.0f), 1.0f);
							float nz = std::sqrtf(1.0f-std::fminf(nx * nx + ny * ny,1.0f));

							pixel_data[(h - 1u - i) * w + j].x = 255u * (0.5f * nx + 0.5f);
							pixel_data[(h - 1u - i) * w + j].y = 255u * (0.5f * ny + 0.5f);
							pixel_data[(h - 1u - i) * w + j].z = 255u * nz;
							pixel_data[(h - 1u - i) * w + j].w = 255u;

							
						}
					}
					{
						auto bamp_path    = filepath.filename().replace_extension();
						auto bamp_path_str = bamp_path.string()+ std::string("_normal.png");
						stbi_write_png(bamp_path_str.c_str(), w, h, 4, pixel_data.data(), 4 * w);
					}
				}
				else {
					for (size_t i = 0; i < h; ++i) {
						for (size_t j = 0; j < w; ++j) {
							if (c == 1) {
								pixel_data[(h - 1u - i) * w + j].x = pixels[1 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].y = pixels[1 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].z = pixels[1 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].w = 255;
							}
							if (c == 2) {
								pixel_data[(h - 1u - i) * w + j].x = pixels[2 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].y = pixels[2 * (i * w + j) + 1];
								pixel_data[(h - 1u - i) * w + j].z = 0;
								pixel_data[(h - 1u - i) * w + j].w = 255;
							}
							if (c == 3) {
								pixel_data[(h - 1u - i) * w + j].x = pixels[3 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].y = pixels[3 * (i * w + j) + 1];
								pixel_data[(h - 1u - i) * w + j].z = pixels[3 * (i * w + j) + 2];
								pixel_data[(h - 1u - i) * w + j].w = 255;
							}
							if (c == 4) {
								pixel_data[(h - 1u - i) * w + j].x = pixels[4 * (i * w + j) + 0];
								pixel_data[(h - 1u - i) * w + j].y = pixels[4 * (i * w + j) + 1];
								pixel_data[(h - 1u - i) * w + j].z = pixels[4 * (i * w + j) + 2];
								pixel_data[(h - 1u - i) * w + j].w = pixels[4 * (i * w + j) + 3];
							}
						}
					}
				}
				auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, w, h, pixel_data.data(), OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
				textures.push_back(tex);

				stbi_image_free(pixels);
			}
		}
	}

	auto module       = owlModuleCreate(context, (const char*)deviceCode_ptx);
	auto accum_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT4, camera.width * camera.height, nullptr);
	auto frame_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, camera.width * camera.height, nullptr);
	auto params       = static_cast<OWLParams>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"                    ,OWLDataType::OWL_GROUP      , offsetof(LaunchParams,world)        },
			OWLVarDecl{"accum_buffer"             ,OWLDataType::OWL_BUFPTR     , offsetof(LaunchParams,accum_buffer) },
			OWLVarDecl{"accum_sample"             ,OWLDataType::OWL_INT        , offsetof(LaunchParams,accum_sample) },
			OWLVarDecl{"frame_buffer"             ,OWLDataType::OWL_BUFPTR     , offsetof(LaunchParams,frame_buffer) },
			OWLVarDecl{"frame_size"               ,OWLDataType::OWL_INT2       , offsetof(LaunchParams,frame_size  ) },
			OWLVarDecl{"light_envmap"             ,OWLDataType::OWL_TEXTURE_2D , offsetof(LaunchParams,light_envmap) },
			OWLVarDecl{"light_envmap_sun"         ,OWLDataType::OWL_FLOAT4     , offsetof(LaunchParams,light_envmap_sun)},
			OWLVarDecl{"light_intensity"          ,OWLDataType::OWL_FLOAT      , offsetof(LaunchParams,light_intensity) },
			OWLVarDecl{"light_parallel.active"    ,OWLDataType::OWL_UINT       , offsetof(LaunchParams,light_parallel)+offsetof(ParallelLight,active)   },
			OWLVarDecl{"light_parallel.color"     ,OWLDataType::OWL_FLOAT3     , offsetof(LaunchParams,light_parallel)+offsetof(ParallelLight,color)    },
			OWLVarDecl{"light_parallel.direction" ,OWLDataType::OWL_FLOAT3     , offsetof(LaunchParams,light_parallel)+offsetof(ParallelLight,direction)},
			OWLVarDecl{nullptr}
		};
		params = owlParamsCreate(context, sizeof(LaunchParams), var_decls, -1);
		owlParamsSetBuffer(params , "accum_buffer"       , accum_buffer);
		owlParamsSet1i(params, "accum_sample", 0);
		owlParamsSetBuffer(params , "frame_buffer"       , frame_buffer);
		owlParamsSet2i(params, "frame_size"  , camera.width, camera.height);
		owlParamsSetTexture(params, "light_envmap"    , textures[texture_idx_envlit]);
		owlParamsSet1f(params, "light_intensity"      , 1.0f);
		owlParamsSet4f(params, "light_envmap_sun"         , envlit_sun_dir.x, envlit_sun_dir.y, envlit_sun_dir.z, envlit_sun_k);
		owlParamsSet1ui(params,"light_parallel.active", 0);
		owlParamsSet3f(params, "light_parallel.color" , 0.0f, 0.0f, 0.0f);
		owlParamsSet3f(params, "light_parallel.direction", 0.0f, 1.0f, 0.0f);
	}

	auto raygen     = static_cast<OWLRayGen>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP  ,offsetof(RayGenData,world)   },
			OWLVarDecl{"min_depth"   ,OWLDataType::OWL_FLOAT  ,offsetof(RayGenData,min_depth)},
			OWLVarDecl{"max_depth"   ,OWLDataType::OWL_FLOAT  ,offsetof(RayGenData,max_depth)},
			OWLVarDecl{"camera.eye"  ,OWLDataType::OWL_FLOAT3 ,offsetof(RayGenData,camera) + offsetof(CameraData,eye)},
			OWLVarDecl{"camera.dir_u",OWLDataType::OWL_FLOAT3 ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_u)},
			OWLVarDecl{"camera.dir_v",OWLDataType::OWL_FLOAT3 ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_v)},
			OWLVarDecl{"camera.dir_w",OWLDataType::OWL_FLOAT3 ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_w)},
			OWLVarDecl{nullptr}
		};
		auto [dir_u, dir_v, dir_w] = camera.getUVW();
		raygen = owlRayGenCreate(context, module, "simpleRG", sizeof(RayGenData), var_decls, -1);
		owlRayGenSet1f(raygen , "min_depth"   , 0.01f);
		owlRayGenSet1f(raygen , "max_depth"   , 2.0f*bbox_len);
		owlRayGenSet3fv(raygen, "camera.eye"  , (const float*)&camera.origin);
		owlRayGenSet3fv(raygen, "camera.dir_u", (const float*)&dir_u);
		owlRayGenSet3fv(raygen, "camera.dir_v", (const float*)&dir_v);
		owlRayGenSet3fv(raygen, "camera.dir_w", (const float*)&dir_w);
	}

	auto miss_prog_radiance  = static_cast<OWLMissProg>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"texture_envlight"  ,OWLDataType::OWL_TEXTURE_2D ,offsetof(MissProgData,texture_envlight)  },
			OWLVarDecl{nullptr}
		};
		miss_prog_radiance  = owlMissProgCreate(context, module, "radianceMS", sizeof(MissProgData), var_decls, -1);
		owlMissProgSetTexture(miss_prog_radiance, "texture_envlight", textures[texture_idx_envlit]);
	}
	auto miss_prog_occluded = static_cast<OWLMissProg>(nullptr);
	{
		miss_prog_occluded  = owlMissProgCreate(context, module, "occludedMS",0, nullptr, 0);
	}

	auto geom_type  = static_cast<OWLGeomType>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"vertices"           ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,vertices) },
			OWLVarDecl{"normals"            ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,normals ) },
			OWLVarDecl{"tangents"           ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,tangents) },
			OWLVarDecl{"uvs"                ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,uvs     ) },
			OWLVarDecl{"colors"             ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,colors  ) },
			OWLVarDecl{"indices"            ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,indices ) },
			OWLVarDecl{"color_ambient"      ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_ambient )  },
			OWLVarDecl{"color_specular"     ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_specular)  },
			OWLVarDecl{"color_emission"     ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_emission)  },
			OWLVarDecl{"shininess"          ,OWLDataType::OWL_FLOAT      ,offsetof(HitgroupData,shininess)       },
			OWLVarDecl{"texture_ambient"    ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_ambient) },
			OWLVarDecl{"texture_normal"     ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_normal)  },
			OWLVarDecl{"texture_specular"   ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_specular)},
			OWLVarDecl{nullptr}
		};
		geom_type  = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES, sizeof(HitgroupData), var_decls, -1);
		owlGeomTypeSetClosestHit(geom_type, RAY_TYPE_RADIANCE, module, "radianceCH");
		owlGeomTypeSetClosestHit(geom_type, RAY_TYPE_OCCLUDED, module, "occludedCH");
	}

	auto geom_type_alpha = static_cast<OWLGeomType>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"vertices"         ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,vertices)         },
			OWLVarDecl{"normals"          ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,normals)          },
			OWLVarDecl{"tangents"         ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,tangents)         },
			OWLVarDecl{"uvs"              ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,uvs)              },
			OWLVarDecl{"colors"           ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,colors)           },
			OWLVarDecl{"indices"          ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,indices)          },
			OWLVarDecl{"color_ambient"    ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_ambient )  },
			OWLVarDecl{"color_specular"   ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_specular)  },
			OWLVarDecl{"color_emission"   ,OWLDataType::OWL_FLOAT3     ,offsetof(HitgroupData,color_emission)  },
			OWLVarDecl{"shininess"        ,OWLDataType::OWL_FLOAT      ,offsetof(HitgroupData,shininess)       },
			OWLVarDecl{"texture_ambient"  ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_ambient) },
			OWLVarDecl{"texture_alpha"    ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_alpha  ) },
			OWLVarDecl{"texture_normal"   ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_normal)  },
			OWLVarDecl{"texture_specular" ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_specular)},
			OWLVarDecl{nullptr}
		};
		geom_type_alpha = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES, sizeof(HitgroupData), var_decls, -1);
		owlGeomTypeSetClosestHit(geom_type_alpha, RAY_TYPE_RADIANCE, module, "radianceCH");
		owlGeomTypeSetClosestHit(geom_type_alpha, RAY_TYPE_OCCLUDED, module, "occludedCH");
		owlGeomTypeSetAnyHit(geom_type_alpha, RAY_TYPE_RADIANCE, module, "simpleAH");
		owlGeomTypeSetAnyHit(geom_type_alpha, RAY_TYPE_OCCLUDED, module, "simpleAH");
	}

	std::unordered_map<std::string, OWLGeom> trim_map = {};
	{
		trim_map.reserve(model.size());
		for (auto& [name, mesh] : model) {
			{
				auto colr_buf = static_cast<OWLBuffer>(nullptr);
				auto vert_buf = static_cast<OWLBuffer>(nullptr);
				auto texc_buf = static_cast<OWLBuffer>(nullptr);
				auto tang_buf = static_cast<OWLBuffer>(nullptr);
				auto norm_buf = static_cast<OWLBuffer>(nullptr);
				auto colors   = mesh.getVisSmoothColors();
				{
					vert_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, mesh.positions.size()  , mesh.positions.data());
					colr_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, colors.size()          , colors.data());
					norm_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, mesh.normals.size()    , mesh.normals.data());
					tang_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT4, mesh.tangents.size()   , mesh.tangents.data());
					texc_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT2, mesh.uvs.size()        , mesh.uvs.data());
				}
				for (auto i = 0; i <     mesh.materials.size(); ++i) {
					auto& material     = model.getMaterials()[mesh.materials[i]];
					auto tri_indices   = mesh.getSubMeshIndices(i);
					auto tmp_geom_type = material.tex_alpha == 0 ? geom_type : geom_type_alpha;
					auto trim          = owlGeomCreate(context, tmp_geom_type);
					auto indx_buf      = owlDeviceBufferCreate(context, OWLDataType::OWL_UINT3, tri_indices.size(), tri_indices.data());
					owlTrianglesSetVertices(trim, vert_buf, owlBufferSizeInBytes(vert_buf) / sizeof(owl::vec3f), sizeof(owl::vec3f), 0);
					owlTrianglesSetIndices(trim, indx_buf , owlBufferSizeInBytes(indx_buf) / sizeof(owl::vec3i), sizeof(owl::vec3i), 0);
					owlGeomSetBuffer(trim, "vertices", vert_buf);
					owlGeomSetBuffer(trim, "colors"  , colr_buf);
					owlGeomSetBuffer(trim, "normals" , norm_buf);
					owlGeomSetBuffer(trim, "tangents", tang_buf);
					owlGeomSetBuffer(trim, "uvs"     , texc_buf);
					owlGeomSetBuffer(trim, "indices" , indx_buf);
					owlGeomSet3f(trim, "color_ambient" , material.diffuse.x , material.diffuse.y , material.diffuse.z );
					owlGeomSet3f(trim, "color_specular", material.specular.x, material.specular.y, material.specular.z);
					owlGeomSet3f(trim, "color_emission", material.emission.x, material.emission.y, material.emission.z);
					owlGeomSet1f(trim, "shininess"     , material.shinness);
					if (material.tex_diffuse == 0) {
						owlGeomSetTexture(trim, "texture_ambient", textures[texture_idx_white]);
					}
					else {
						owlGeomSetTexture(trim, "texture_ambient", textures[material.tex_diffuse + texture_idx_offset]);
					}
					if (material.tex_specular == 0) {
						owlGeomSetTexture(trim, "texture_specular", textures[texture_idx_white]);
					}
					else {
						owlGeomSetTexture(trim, "texture_specular", textures[material.tex_specular + texture_idx_offset]);
					}
					if (material.tex_normal    == 0) {
						owlGeomSetTexture(trim, "texture_normal", textures[texture_idx_blue]);
					}
					else {
						owlGeomSetTexture(trim, "texture_normal", textures[material.tex_normal  + texture_idx_offset]);
					}
					if (material.tex_alpha != 0) {
						owlGeomSetTexture(trim, "texture_alpha"  , textures[material.tex_alpha  + texture_idx_offset]);
					}
					trim_map.insert({ name + "[" + std::to_string(i) + "]", trim});
				}
			}
		}
	}
	auto trim_group = static_cast<OWLGroup>(nullptr);
	{
		std::vector<OWLGeom> trim_arr = {};
		trim_arr.reserve(trim_map.size() );
		for (auto& [name, geom] : trim_map) { trim_arr.push_back(geom); }

		trim_group  = owlTrianglesGeomGroupCreate(context, trim_arr.size(), trim_arr.data(), 0);
		owlGroupBuildAccel(trim_group);
	}

	auto world      = static_cast<OWLGroup>(nullptr);
	{
		world       = owlInstanceGroupCreate(context, 1, &trim_group);
		owlGroupBuildAccel(world);
		owlParamsSetGroup(params, "world", world);
		owlRayGenSetGroup(raygen, "world", world);
	}

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context, (OWLBuildSBTFlags)(OWLBuildSBTFlags::OWL_SBT_ALL2));

	{
		auto tonemap = hikari::test::owl::testlib::Tonemap(camera.width, camera.height, 0.104f);
		tonemap.init();
		struct TracerData {
			hikari::test::owl::testlib::PinholeCamera* p_camera;
			hikari::test::owl::testlib::Tonemap*       p_tonemap;
			OWLContext                                 context;
			OWLRayGen                                  raygen;
			OWLParams                                  params;
			OWLBuffer                                  accum_buffer;
			OWLBuffer                                  frame_buffer;
			int                                        accum_sample;
			bool                                       estimate_luminance;
			bool                                       use_parallel_light;
			int                                        light_type;
			float                                      env_light_intensity;
			float4                                     env_light_sun;
			float3                                     parallel_light_color;
			float                                      parallel_light_intensity;
			float                                      parallel_light_axis_phi;
			float                                      parallel_light_axis_tht;
			float                                      parallel_light_angle;
		} tracer_data = {
			 &camera,& tonemap,context, raygen,params,accum_buffer,frame_buffer, 0,true, false,LIGHT_TYPE_ENVMAP,1.0f,{envlit_sun_dir.x,envlit_sun_dir.y,envlit_sun_dir.z,envlit_sun_k},{0.0f,0.0f,0.0f},1.0f,0.0f,0.0f
		};
		 
		auto resize_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, int old_w, int old_h, int new_w, int        new_h) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			owlBufferResize(p_tracer_data->accum_buffer, new_w * new_h);
			owlBufferResize(p_tracer_data->frame_buffer, new_w * new_h);
			owlParamsSet2i(p_tracer_data->params, "frame_size", new_w, new_h);
			p_tracer_data->p_camera->width = new_w;
			p_tracer_data->p_camera->height = new_h;
			p_tracer_data->p_tonemap->resize(new_w, new_h);
			return true;
			};
		auto presskey_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, hikari::test::owl::testlib::KeyType           key) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			if (key == hikari::test::owl::testlib::KeyType::eW) { p_tracer_data->p_camera->processPressKeyW(1.0f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eS) { p_tracer_data->p_camera->processPressKeyS(1.0f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eA) { p_tracer_data->p_camera->processPressKeyA(1.0f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eD) { p_tracer_data->p_camera->processPressKeyD(1.0f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eLeft) { p_tracer_data->p_camera->processPressKeyLeft(0.5f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eRight) { p_tracer_data->p_camera->processPressKeyRight(0.5f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eUp) { p_tracer_data->p_camera->processPressKeyUp(0.5f); return true; }
			if (key == hikari::test::owl::testlib::KeyType::eDown) { p_tracer_data->p_camera->processPressKeyDown(0.5f); return true; }
			return false;
			};
		auto press_mouse_button_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, hikari::test::owl::testlib::MouseButtonType mouse) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			if (mouse == hikari::test::owl::testlib::MouseButtonType::eLeft) {
				int width; int height;
				double x; double y;
				p_viewer->getCursorPosition(x, y);
				p_viewer->getWindowSize(width, height);
				// ���オ(0,0), �E����(1,1)
				float sx = std::clamp((float)x / (float)width, 0.0f, 1.0f);
				float sy = std::clamp((float)y / (float)height, 0.0f, 1.0f);
				printf("%f %f\n", sx, sy);
				if (sx < 0.5f) { p_tracer_data->p_camera->processPressKeyLeft(0.5f - sx); }
				else { p_tracer_data->p_camera->processPressKeyRight(sx - 0.5f); }
				if (sy < 0.5f) { p_tracer_data->p_camera->processPressKeyUp(0.5f - sy); }
				else { p_tracer_data->p_camera->processPressKeyDown(sy - 0.5f); }
				return true;
			}
			return false;
			};
		auto mouse_scroll_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, double x, double y) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			if (y != 0.0f) {
				p_tracer_data->p_camera->processMouseScrollY(y);
				return true;
			}
			return false;
			};
		auto update_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			auto [dir_u, dir_v, dir_w] = p_tracer_data->p_camera->getUVW();
			owlBufferClear(p_tracer_data->accum_buffer);
			owlBufferClear(p_tracer_data->frame_buffer);
			owlRayGenSet3fv(p_tracer_data->raygen, "camera.eye", (const float*)&p_tracer_data->p_camera->origin);
			owlRayGenSet3fv(p_tracer_data->raygen, "camera.dir_u", (const float*)&dir_u);
			owlRayGenSet3fv(p_tracer_data->raygen, "camera.dir_v", (const float*)&dir_v);
			owlRayGenSet3fv(p_tracer_data->raygen, "camera.dir_w", (const float*)&dir_w);
			owlParamsSet1f(p_tracer_data->params, "light_intensity", p_tracer_data->env_light_intensity);
			owlParamsSet1ui(p_tracer_data->params, "light_parallel.active", p_tracer_data->use_parallel_light);
			owlParamsSet3f(p_tracer_data->params, "light_parallel.color",
				p_tracer_data->parallel_light_color.x * p_tracer_data->parallel_light_intensity,
				p_tracer_data->parallel_light_color.y * p_tracer_data->parallel_light_intensity,
				p_tracer_data->parallel_light_color.z * p_tracer_data->parallel_light_intensity
			);
			owlParamsSet4f(p_tracer_data->params, "light_envmap_sun", p_tracer_data->env_light_sun.x, p_tracer_data->env_light_sun.y, p_tracer_data->env_light_sun.z, p_tracer_data->env_light_sun.w);
			auto cos_phi = cosf(p_tracer_data->parallel_light_axis_phi * M_PI / 180.0f);
			auto sin_phi = sinf(p_tracer_data->parallel_light_axis_phi * M_PI / 180.0f);
			auto cos_tht = cosf(p_tracer_data->parallel_light_axis_tht * M_PI / 180.0f);
			auto sin_tht = sinf(p_tracer_data->parallel_light_axis_tht * M_PI / 180.0f);
			auto cos_ang = cosf(p_tracer_data->parallel_light_angle * M_PI / 180.0f);
			auto sin_ang = sinf(p_tracer_data->parallel_light_angle * M_PI / 180.0f);
			// y * z -z * y  = -sin_phi
			// z * x -x * z  = 0.0
			// x * y -y * x  = cos_phi
			auto light_axis_u = owl::vec3f(-sin_phi, 0.0f, cos_phi);
			auto light_axis_v = owl::vec3f(sin_tht * cos_phi, cos_tht, sin_tht * sin_phi);
			auto light_dir = cos_ang * light_axis_u + sin_ang * light_axis_v;
			owlParamsSet3f(p_tracer_data->params, "light_parallel.direction",
				light_dir.x,
				light_dir.y,
				light_dir.z
			);

			p_tracer_data->accum_sample = 0;
			};
		auto render_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer, void* p_fb_data) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			owlParamsSet1i(p_tracer_data->params, "accum_sample", p_tracer_data->accum_sample);
			owlBuildSBT(p_tracer_data->context, (OWLBuildSBTFlags)(OWL_SBT_RAYGENS | OWL_SBT_CALLABLES));
			owlLaunch2D(p_tracer_data->raygen, p_tracer_data->p_camera->width, p_tracer_data->p_camera->height, p_tracer_data->params);
			p_tracer_data->p_tonemap->launch(owlContextGetStream(p_tracer_data->context, 0),
				(const float3*)owlBufferGetPointer(p_tracer_data->frame_buffer, 0),
				(unsigned int*)p_fb_data,
				!p_tracer_data->estimate_luminance
			);
			p_tracer_data->accum_sample++;
			};
		auto ui_callback = [](hikari::test::owl::testlib::GLViewer* p_viewer) {
			TracerData* p_tracer_data = (TracerData*)p_viewer->getUserPtr();
			if (ImGui::Begin("Config")) {
				if (ImGui::TreeNode("Tonemap")) {
					float new_key_value = p_tracer_data->p_tonemap->getKeyValue();
					float old_key_value = new_key_value;
					{
						const char* combo_defaults[] = { "Linear","Linear(Correlated)","Reinhard(Correlated","Extended Reinhard(Correlated)" };
						if (ImGui::BeginCombo("Type", combo_defaults[(int)p_tracer_data->p_tonemap->getType()])) {
							if (ImGui::Selectable("Linear")) {
								p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eLinear);
							}
							if (ImGui::Selectable("Linear(Correlated)")) {
								p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedLinear);
							}
							if (ImGui::Selectable("Reinhard(Correlated)")) {
								p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedReinhard);
							}
							if (ImGui::Selectable("Extended Reinhard(Correlated)")) {
								p_tracer_data->p_tonemap->setType(hikari::test::owl::testlib::TonemapType::eCorrelatedExtendedReinhard);
							}
							ImGui::EndCombo();
						}
					}
					ImGui::SliderFloat("Key Value: ", &new_key_value, 0.001f, 5.0f);
					ImGui::Text("Maxmimum Luminance: %f", p_tracer_data->p_tonemap->getMaxLuminance());
					ImGui::Text("Average  Luminance: %f", p_tracer_data->p_tonemap->getAveLuminance());
					if (new_key_value != old_key_value) {
						p_tracer_data->p_tonemap->setKeyValue(new_key_value);
					}
					bool v = p_tracer_data->estimate_luminance;
					if (ImGui::Checkbox("Estimate Luminance", &v)) {
						p_tracer_data->estimate_luminance = v;
					}
					ImGui::TreePop();
				}
				if (ImGui::TreeNode("Camera")) {
					float camera_pos[3] = { p_tracer_data->p_camera->origin.x, p_tracer_data->p_camera->origin.y, p_tracer_data->p_camera->origin.z };
					if (ImGui::InputFloat3("Position: ", camera_pos)) {
						p_tracer_data->p_camera->origin.x = camera_pos[0];
						p_tracer_data->p_camera->origin.y = camera_pos[1];
						p_tracer_data->p_camera->origin.z = camera_pos[2];
						p_viewer->updateNextFrame();
					}
					float camera_dir[3] = { p_tracer_data->p_camera->direction.x, p_tracer_data->p_camera->direction.y, p_tracer_data->p_camera->direction.z };
					if (ImGui::InputFloat3("Direction: ", camera_dir)) {
						if (camera_dir[0] != 0.0f || camera_dir[1] != 0.0f || camera_dir[2] != 0.0f) {
							p_tracer_data->p_camera->direction.x = camera_dir[0];
							p_tracer_data->p_camera->direction.y = camera_dir[1];
							p_tracer_data->p_camera->direction.z = camera_dir[2];
							p_tracer_data->p_camera->direction = glm::normalize(p_tracer_data->p_camera->direction);
							p_viewer->updateNextFrame();
						}
					}
					float camera_speed[3] = { p_tracer_data->p_camera->speed.x, p_tracer_data->p_camera->speed.y, p_tracer_data->p_camera->speed.z };
					if (ImGui::InputFloat3("Speed: ", camera_speed)) {
						p_tracer_data->p_camera->speed.x = camera_speed[0];
						p_tracer_data->p_camera->speed.y = camera_speed[1];
						p_tracer_data->p_camera->speed.z = camera_speed[2];
					}
					float camera_fovy = p_tracer_data->p_camera->fovy;
					if (ImGui::SliderFloat("FovY: ", &camera_fovy, 0.0f, 180.0f)) {
						p_tracer_data->p_camera->fovy = camera_fovy;
						p_viewer->updateNextFrame();
					}
					ImGui::TreePop();
				}
				if (ImGui::TreeNode("Light")) {
					const char* combo_defaults[] = { "EnvMap","Directional" };
					if (ImGui::BeginCombo("Type", combo_defaults[(int)p_tracer_data->light_type])) {
						if (ImGui::Selectable("EnvMap")) {
							if (p_tracer_data->light_type != LIGHT_TYPE_ENVMAP) {
								p_tracer_data->light_type = LIGHT_TYPE_ENVMAP;
							}
						}
						if (ImGui::Selectable("Directional")) {
							if (p_tracer_data->light_type != LIGHT_TYPE_DIRECTIONAL) {
								p_tracer_data->light_type = LIGHT_TYPE_DIRECTIONAL;
							}

						}
						ImGui::EndCombo();
					}
					if (p_tracer_data->light_type == LIGHT_TYPE_ENVMAP) {
						float intensity = p_tracer_data->env_light_intensity;
						if (ImGui::SliderFloat("intensity", &intensity, 0.0f, 10.0f)) {
							p_tracer_data->env_light_intensity = intensity;
							p_viewer->updateNextFrame();
						}
						float env_light_sun_k = p_tracer_data->env_light_sun.w;
						if (ImGui::SliderFloat("sun_k", &env_light_sun_k, 0.0f, 10.0f)) {
							p_tracer_data->env_light_sun.w = env_light_sun_k;
							p_viewer->updateNextFrame();
						}
					}
					if (p_tracer_data->light_type == LIGHT_TYPE_DIRECTIONAL) {
						float color[3] = { p_tracer_data->parallel_light_color.x,p_tracer_data->parallel_light_color.y,p_tracer_data->parallel_light_color.z };
						if (ImGui::ColorPicker3("color", color)) {
							p_tracer_data->parallel_light_color.x = color[0];
							p_tracer_data->parallel_light_color.y = color[1];
							p_tracer_data->parallel_light_color.z = color[2];
							p_viewer->updateNextFrame();
						}
						float intensity = p_tracer_data->parallel_light_intensity;
						if (ImGui::SliderFloat("intensity", &intensity, 0.0f, 10.0f)) {
							p_tracer_data->parallel_light_intensity = intensity;
							p_viewer->updateNextFrame();
						}
						float phi = p_tracer_data->parallel_light_axis_phi;
						float tht = p_tracer_data->parallel_light_axis_tht;
						float ang = p_tracer_data->parallel_light_angle;
						if (ImGui::SliderFloat("axis(phi)", &phi, 0.0f, 360.0f)) {
							p_tracer_data->parallel_light_axis_phi = phi;
							p_viewer->updateNextFrame();
						}
						if (ImGui::SliderFloat("axis(tht)", &tht, 0.0f, 90.0f)) {
							p_tracer_data->parallel_light_axis_tht = tht;
							p_viewer->updateNextFrame();
						}
						if (ImGui::SliderFloat("angle", &ang, 0.0f, 180.0f)) {
							p_tracer_data->parallel_light_angle = ang;
							p_viewer->updateNextFrame();
						}


						if (ImGui::Checkbox("active", &p_tracer_data->use_parallel_light)) {
							p_viewer->updateNextFrame();
						}
					}
					ImGui::TreePop();
				}
			}
			ImGui::End();
			};
		auto viewer                      = std::make_unique<hikari::test::owl::testlib::GLViewer>(owlContextGetStream(context,0), camera.width, camera.height);
		viewer->runWithCallback(&tracer_data, resize_callback, presskey_callback, press_mouse_button_callback, mouse_scroll_callback, update_callback, render_callback, ui_callback);
		viewer.reset();
		tonemap.free();
	}
	return 0;
}