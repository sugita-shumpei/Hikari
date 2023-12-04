#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <string>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <tuple>
#include <random>
#include <filesystem>
#include "hostCode.h"
#include "deviceCode.h"
#include "kernelCode.h"
#include "thrustCode.h"

extern "C" char* deviceCode_ptx[];
extern "C" char* kernelCode_ptx[];

int main() {
	hikari::test::owl::testlib::ObjModel model;
	model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\CornellBox\CornellBox-Original.obj)");
    //model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Sponza\sponza.obj)");
	//model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Bistro\Exterior\exterior.obj)");/*
	//auto envlit_filename = std::string(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Textures\evening_road_01_puresky_8k.hdr)");*/
    auto envlit_filename = std::string("");
	auto center          = model.getBBox().getCenter();
	auto range           = model.getBBox().getRange ();
	hikari::test::owl::testlib::PinholeCamera camera;
	camera.origin.x      = center.x;
	camera.origin.y      = center.y;
	camera.origin.z      =  5.0f;
	camera.direction.z   = -1.0f;
	camera.speed.x       = range.x * 0.01f / 2.0f;
	camera.speed.y       = range.z * 0.01f / 2.0f;

	auto bbox_len = 2.0f*std::sqrtf(range.x * range.x + range.y * range.y + range.z * range.z);
	auto context  = owlContextCreate();
	owlContextSetRayTypeCount(context, RAY_TYPE_COUNT);
	auto textures = std::vector<OWLTexture>();

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
			auto pixels = stbi_loadf(envlit_filename.data(), &w, &h, &c, 0);
			assert(pixels);
			auto pixel_data = std::vector<owl::vec4f>();
			pixel_data.resize(w * h, { 0.0f,0.0f,0.0f,0.0f });
			{
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
					}
				}
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
			OWLVarDecl{"world"           ,OWLDataType::OWL_GROUP      , offsetof(LaunchParams,world)       },
			OWLVarDecl{"texture_envlight",OWLDataType::OWL_TEXTURE_2D , offsetof(LaunchParams,texture_envlight)},
			OWLVarDecl{"accum_buffer"    ,OWLDataType::OWL_BUFPTR     , offsetof(LaunchParams,accum_buffer)},
			OWLVarDecl{"frame_buffer"    ,OWLDataType::OWL_BUFPTR     , offsetof(LaunchParams,frame_buffer)},
			OWLVarDecl{"frame_size"      ,OWLDataType::OWL_INT2       , offsetof(LaunchParams,frame_size  )},
			OWLVarDecl{"accum_sample"    ,OWLDataType::OWL_INT        , offsetof(LaunchParams,accum_sample)},
			OWLVarDecl{nullptr}
		};
		params = owlParamsCreate(context, sizeof(LaunchParams), var_decls, -1);
		owlParamsSetBuffer(params , "accum_buffer"    , accum_buffer);
		owlParamsSetBuffer(params , "frame_buffer"    , frame_buffer);
		owlParamsSetTexture(params, "texture_envlight", textures[texture_idx_envlit]);
		owlParamsSet2i(params, "frame_size"  , camera.width, camera.height);
		owlParamsSet1i(params, "accum_sample", 0);
	}

	auto raygen     = static_cast<OWLRayGen>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP      ,offsetof(RayGenData,world)   },
			OWLVarDecl{"min_depth"   ,OWLDataType::OWL_FLOAT      ,offsetof(RayGenData,min_depth)},
			OWLVarDecl{"max_depth"   ,OWLDataType::OWL_FLOAT      ,offsetof(RayGenData,max_depth)},
			OWLVarDecl{"camera.eye"  ,OWLDataType::OWL_FLOAT3     ,offsetof(RayGenData,camera) + offsetof(CameraData,eye)},
			OWLVarDecl{"camera.dir_u",OWLDataType::OWL_FLOAT3     ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_u)},
			OWLVarDecl{"camera.dir_v",OWLDataType::OWL_FLOAT3     ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_v)},
			OWLVarDecl{"camera.dir_w",OWLDataType::OWL_FLOAT3     ,offsetof(RayGenData,camera) + offsetof(CameraData,dir_w)},
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
				for (auto i = 0; i < mesh.materials.size(); ++i) {
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

	auto callable  = static_cast<OWLCallable>(nullptr);
	{
		OWLVarDecl varDecls[] = {
			OWLVarDecl{"color", OWLDataType::OWL_FLOAT4, offsetof(CallableData,color)},
			OWLVarDecl{nullptr}
		};

		callable = owlCallableCreate(context, module, "simpleDC1", true, sizeof(CallableData), varDecls, -1);
		owlCallableSet4f(callable, "color", owl4f(1.0f, 0.0f, 0.0f, 1.0f));

		callable = owlCallableCreate(context, module, "simpleDC2", true, sizeof(CallableData), varDecls, -1);
		owlCallableSet4f(callable, "color", owl4f(0.0f, 1.0f, 0.0f, 1.0f));
	}

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context, (OWLBuildSBTFlags)(OWLBuildSBTFlags::OWL_SBT_ALL2));

	auto module_kernel = CUmodule();
	auto kernel_tonemap             = CUfunction(nullptr);
	auto kernel_convertToRGBA8      = CUfunction(nullptr);
	auto kernel_convertToLuminance  = CUfunction(nullptr);
	auto frame_luminance_buffer     = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT, camera.width * camera.height, nullptr);
	auto frame_luminance_log_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT, camera.width * camera.height, nullptr);
	{
		cuModuleLoadData(&module_kernel, kernelCode_ptx);
		cuModuleGetFunction(&kernel_tonemap           , module_kernel, "__kernel__tonemap"           );
		cuModuleGetFunction(&kernel_convertToRGBA8    , module_kernel, "__kernel__convertToRGBA8"    );
		cuModuleGetFunction(&kernel_convertToLuminance, module_kernel, "__kernel__convertToLuminance");
	}
	{
		glfwInit();
		glfwWindowHint(GLFW_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* window = glfwCreateWindow(camera.width, camera.height, "title", nullptr, nullptr);
		glfwMakeContextCurrent(window);
		if (!hikari::test::owl::testlib::loadGLLoader((hikari::test::owl::testlib::GLloadproc)glfwGetProcAddress)){
			return -1;
		}
		auto viewer = std::make_unique<hikari::test::owl::testlib::GLViewer>(context, camera.width, camera.height);
		{
			int accum_sample = 0;
			glfwShowWindow(window);
			while (!glfwWindowShouldClose(window)) {
				{
					glfwPollEvents();
					glfwGetWindowSize(window, &camera.width, &camera.height);
					bool update = false;
					if (viewer->resize(camera.width, camera.height)) {
						printf("%d %d\n", camera.width, camera.height);
						owlBufferResize(accum_buffer          , camera.width * camera.height);
						owlBufferResize(frame_buffer          , camera.width * camera.height);
						owlBufferResize(frame_luminance_buffer, camera.width * camera.height);
						owlBufferResize(frame_luminance_log_buffer, camera.width* camera.height);
						owlParamsSet2i(params, "frame_size", camera.width, camera.height);
						update = true;
					}
					{
						if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
							double x; double y;
							glfwGetCursorPos(window, &x, &y);
							// ���オ(0,0), �E����(1,1)
							float sx = std::clamp((float)x / (float)camera.width, 0.0f, 1.0f);
							float sy = std::clamp((float)y / (float)camera.height, 0.0f, 1.0f);
							printf("%f %f\n", sx, sy);
							if (sx < 0.5f) {
								camera.processPressKeyLeft(0.5f - sx);
							}
							else {
								camera.processPressKeyRight(sx - 0.5f);
							}
							if (sy < 0.5f) {
								camera.processPressKeyUp(0.5f - sy);
							}
							else {
								camera.processPressKeyDown(sy - 0.5f);
							}
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_W)     == GLFW_PRESS) {
							camera.processPressKeyW(1.0f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_S)     == GLFW_PRESS) {
							camera.processPressKeyS(1.0f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_A)     == GLFW_PRESS) {
							camera.processPressKeyA(1.0f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_D)     == GLFW_PRESS) {
							camera.processPressKeyD(1.0f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) {
							camera.processPressKeyUp(0.5f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) {
							camera.processPressKeyDown(0.5f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) {
							camera.processPressKeyLeft(0.5f);
							update = true;
						}
						if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
							camera.processPressKeyRight(0.5f);
							update = true;
						}
					}
					if (update) {
						auto [dir_u, dir_v, dir_w] = camera.getUVW();
						owlBufferClear(accum_buffer);
						owlBufferClear(frame_buffer);
						owlRayGenSet3fv(raygen, "camera.eye"  , (const float*)&camera.origin);
						owlRayGenSet3fv(raygen, "camera.dir_u", (const float*)&dir_u);
						owlRayGenSet3fv(raygen, "camera.dir_v", (const float*)&dir_v);
						owlRayGenSet3fv(raygen, "camera.dir_w", (const float*)&dir_w);
						accum_sample = 0;
					}
					else {
						accum_sample++;
					}
					owlParamsSet1i(params, "accum_sample", accum_sample);
				}
			
				//owlRayGenSetPointer(raygen, "fb_data" , viewer->mapFramePtr());
				owlBuildSBT(context, OWL_SBT_RAYGENS);
				owlLaunch2D(raygen, camera.width, camera.height, params);
				{
					// Luminance用kernel
					int width = camera.width; int height = camera.height;
					auto color_buf     = reinterpret_cast<CUdeviceptr>(owlBufferGetPointer(frame_buffer              , 0));
					auto lumin_buf     = reinterpret_cast<CUdeviceptr>(owlBufferGetPointer(frame_luminance_buffer    , 0));
					auto lumin_log_buf = reinterpret_cast<CUdeviceptr>(owlBufferGetPointer(frame_luminance_log_buffer, 0));
					// 描画用kernel
					void* args[] = {
						&width, &height, & color_buf, & lumin_buf,& lumin_log_buf
					};
					auto grid_x = (width  + 32u - 1u) / 32u;
					auto grid_y = (height + 32u - 1u) / 32u;
					cuLaunchKernel(
						kernel_convertToLuminance,
						grid_x, grid_y, 1,
						32, 32, 1,
						0,
						owlContextGetStream(context, 0),
						args,
						nullptr
					);
					cuStreamSynchronize(owlContextGetStream(context, 0));
				}
				// 平均値と最大値を求める
				float max_v; float log_average_v;
				calculateLogAverageAndMax(camera.width, camera.height,
					(float*)owlBufferGetPointer(frame_luminance_buffer    , 0), 
					(float*)owlBufferGetPointer(frame_luminance_log_buffer, 0),
					&max_v,
					&log_average_v
				);
				{
					int width   = camera.width; int height = camera.height;
					auto pixel3fs = reinterpret_cast<CUdeviceptr>(owlBufferGetPointer(frame_buffer, 0));
					auto pixel32s = reinterpret_cast<CUdeviceptr>(viewer->mapFramePtr());
					float key_value = 0.18f;
					// 描画用kernel
					void* args[] = {
						&width, &height,&key_value,& log_average_v,
						&max_v,
						&pixel3fs, & pixel32s
					};
					auto grid_x = (width  + 32u - 1u) / 32u;
					auto grid_y = (height + 32u - 1u) / 32u;
					cuLaunchKernel(
						kernel_tonemap,
						grid_x, grid_y,1,
						32,32,1,
						0,
						owlContextGetStream(context,0),
						args,
						nullptr
					);
					cuStreamSynchronize(owlContextGetStream(context, 0));
					viewer->unmapFramePtr();
				}
				viewer->render();
				glfwSwapBuffers(window);
			}
		}
		viewer.reset();
		glfwHideWindow(window);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	cuModuleUnload(module_kernel);

	return 0;
}