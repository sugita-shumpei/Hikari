#define STB_IMAGE_IMPLEMENTATION 
#include <iostream>
#include <string>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>
#include <stb_image.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <tuple>
#include <random>
#include <filesystem>
#include "hostCode.h"
#include "deviceCode.h"

extern "C" char* deviceCode_ptx[];


int main() {
	hikari::test::owl::testlib::PinholeCamera camera;
	camera.origin.y    =  1.0f;
	camera.origin.z    =  3.0f;
	camera.direction.z = -1.0f;

	hikari::test::owl::testlib::ObjModel model;
	//model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\CornellBox\CornellBox-Original.obj)");
	model.setFilename(R"(D:\Users\shumpei\Document\Github\RTLib\Data\Models\Sponza\sponza.obj)");
	auto center = model.getBBox().getCenter();
	auto range  = model.getBBox().getRange ();
	
	camera.speed.x = range.x * 0.01f / 2.0f;
	camera.speed.y = range.z * 0.01f / 2.0f;

	auto bbox_len     = 2.0f *std::sqrtf(range.x * range.x + range.y * range.y + range.z * range.z);
	auto context      = owlContextCreate();
	auto textures     = std::vector<OWLTexture>();

	{
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
		// model
		{
			for (auto& texture : model.getTextures()) {
				if (texture.filename == "") { continue; }
				auto filepath     = std::filesystem::canonical(std::filesystem::path(model.getFilename()).parent_path() / texture.filename);
				auto filepath_str = filepath.string();
				int w, h, c;
				auto pixels      = stbi_load(filepath_str.data(), &w, &h, &c, 0);
				assert(pixels);
				auto pixel_data = std::vector<owl::vec4uc>();
				pixel_data.resize(w * h );
				for (size_t i = 0; i < h; ++i) {
					for (size_t j = 0; j < w; ++j) {
						if (c == 1) {
							pixel_data[(h - 1u - i) * w + j].x     = pixels[1 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = 255;
							pixel_data[(h - 1u - i) * w + j].z = 255;
							pixel_data[(h - 1u - i) * w + j].w = 255;
						}
						if (c == 2) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[2 * (i * w + j) + 0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[2 * (i * w + j) + 1];
							pixel_data[(h - 1u - i) * w + j].z = 255;
							pixel_data[(h - 1u - i) * w + j].w = 255;
						}
						if (c == 3) {
							pixel_data[(h - 1u - i) * w + j].x = pixels[3 * (i * w + j) +0];
							pixel_data[(h - 1u - i) * w + j].y = pixels[3 * (i * w + j) +1];
							pixel_data[(h - 1u - i) * w + j].z = pixels[3 * (i * w + j) +2];
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
				auto tex = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA8, w, h, pixel_data.data());
				textures.push_back(tex);
				stbi_image_free(pixels);
			}
		}
	}

	auto module       = owlModuleCreate(context, (const char*)deviceCode_ptx);
	auto accum_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT4, camera.width * camera.height, nullptr);
	auto params       = static_cast<OWLParams>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP , offsetof(LaunchParams,world)       },
			OWLVarDecl{"accum_buffer",OWLDataType::OWL_BUFPTR, offsetof(LaunchParams,accum_buffer)},
			OWLVarDecl{"accum_sample",OWLDataType::OWL_INT   , offsetof(LaunchParams,accum_sample)},
			OWLVarDecl{nullptr}
		};
		params = owlParamsCreate(context, sizeof(LaunchParams), var_decls, -1);
		owlParamsSetBuffer(params, "accum_buffer", accum_buffer);
		owlParamsSet1i(params, "accum_sample", 0);
	}


	auto raygen     = static_cast<OWLRayGen>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP      ,offsetof(RayGenData,world)},
			OWLVarDecl{"fb_data"     ,OWLDataType::OWL_RAW_POINTER,offsetof(RayGenData,fb_data)},
			OWLVarDecl{"fb_size"     ,OWLDataType::OWL_INT2       ,offsetof(RayGenData,fb_size)},
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
		owlRayGenSetPointer(raygen            , "fb_data", nullptr);
		owlRayGenSet2i(raygen , "fb_size"     , camera.width, camera.height);
		owlRayGenSet1f(raygen , "min_depth"   , 0.001f);
		owlRayGenSet1f(raygen , "max_depth"   , bbox_len);
		owlRayGenSet3fv(raygen, "camera.eye"  , (const float*)&camera.origin);
		owlRayGenSet3fv(raygen, "camera.dir_u", (const float*)&dir_u);
		owlRayGenSet3fv(raygen, "camera.dir_v", (const float*)&dir_v);
		owlRayGenSet3fv(raygen, "camera.dir_w", (const float*)&dir_w);
	}

	auto miss_prog  = static_cast<OWLMissProg>(nullptr);
	{
		miss_prog = owlMissProgCreate(context, module, "simpleMS", sizeof(MissProgData), nullptr, 0);
	}

	auto geom_type  = static_cast<OWLGeomType>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"vertices" ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,vertices) },
			OWLVarDecl{"normals"  ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,normals ) },
			OWLVarDecl{"uvs"      ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,uvs     ) },
			OWLVarDecl{"colors"   ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,colors  ) },
			OWLVarDecl{"indices"  ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,indices ) },
			OWLVarDecl{"texture_ambient"  ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_ambient)  },
			OWLVarDecl{nullptr}
		};
		geom_type  = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES, sizeof(HitgroupData), var_decls, -1);
		owlGeomTypeSetClosestHit(geom_type, 0, module, "simpleCH");
	}
	auto geom_type_alpha = static_cast<OWLGeomType>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"vertices"         ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,vertices) },
			OWLVarDecl{"normals"          ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,normals) },
			OWLVarDecl{"uvs"              ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,uvs) },
			OWLVarDecl{"colors"           ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,colors) },
			OWLVarDecl{"indices"          ,OWLDataType::OWL_BUFPTR     ,offsetof(HitgroupData,indices) },
			OWLVarDecl{"texture_ambient"  ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_ambient)  },
			OWLVarDecl{"texture_alpha"    ,OWLDataType::OWL_TEXTURE_2D ,offsetof(HitgroupData,texture_alpha  )  },
			OWLVarDecl{nullptr}
		};
		geom_type_alpha = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES, sizeof(HitgroupData), var_decls, -1);
		owlGeomTypeSetClosestHit(geom_type_alpha, 0, module, "simpleCH");
		owlGeomTypeSetAnyHit(geom_type_alpha, 0, module, "simpleAH");
	}

	std::unordered_map<std::string, OWLGeom> trim_map = {};
	{
		trim_map.reserve(model.size());
		for (auto& [name, mesh] : model) {
			{
				auto colr_buf = static_cast<OWLBuffer>(nullptr);
				auto vert_buf = static_cast<OWLBuffer>(nullptr);
				auto texc_buf = static_cast<OWLBuffer>(nullptr);
				auto norm_buf = static_cast<OWLBuffer>(nullptr);

				auto colors = mesh.getVisSmoothColors();
				{
					vert_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, mesh.positions.size()  , mesh.positions.data());
					colr_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, colors.size()          , colors.data());
					norm_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, mesh.normals.size()    , mesh.normals.data());
					texc_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT2, mesh.uvs.size()        , mesh.uvs.data());
				}

				for (auto i = 0; i < mesh.materials.size(); ++i) {
					auto& material     = model.getMaterials()[mesh.materials[i]];
					auto tri_indices   = mesh.getSubMeshIndices(i);
					auto tmp_geom_type = material.tex_alpha == 0 ? geom_type : geom_type_alpha;
					auto trim          = owlGeomCreate(context, tmp_geom_type);
					auto indx_buf      = owlDeviceBufferCreate(context, OWLDataType::OWL_UINT3, tri_indices.size(), tri_indices.data());
					owlTrianglesSetVertices(trim, vert_buf, owlBufferSizeInBytes(vert_buf) / sizeof(owl::vec3f), sizeof(owl::vec3f), 0);
					owlTrianglesSetIndices(trim, indx_buf, owlBufferSizeInBytes(indx_buf) / sizeof(owl::vec3i), sizeof(owl::vec3i), 0);
					owlGeomSetBuffer(trim, "vertices", vert_buf);
					owlGeomSetBuffer(trim, "colors"  , colr_buf);
					owlGeomSetBuffer(trim, "normals" , norm_buf);
					owlGeomSetBuffer(trim, "uvs"     , texc_buf);
					owlGeomSetBuffer(trim, "indices" , indx_buf);
					if (material.tex_diffuse == 0) {
						owlGeomSetTexture(trim, "texture_ambient", textures[0]);
					}
					else {
						owlGeomSetTexture(trim, "texture_ambient", textures[material.tex_diffuse + 1]);
					}
					if (material.tex_alpha != 0) {
						owlGeomSetTexture(trim, "texture_alpha"  , textures[material.tex_alpha  + 1]);
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
		glfwShowWindow(window);
		while (!glfwWindowShouldClose(window)) {
			{
				glfwPollEvents();
				glfwGetWindowSize(window, &camera.width, &camera.height);
				if (viewer->resize(camera.width, camera.height)) {
					printf("%d %d\n", camera.width, camera.height);
					owlBufferResize(accum_buffer, camera.width * camera.height);
					owlParamsSetBuffer(params, "accum_buffer", accum_buffer);
					owlParamsSet1i(params, "accum_sample", 0);
					owlRayGenSet2i(raygen, "fb_size", camera.width, camera.height);
				}
				{
					bool update = false;
					if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
						double x; double y;
						glfwGetCursorPos(window, &x, &y);
						// ç∂è„Ç™(0,0), âEâ∫Ç™(1,1)
						float sx = std::clamp((float)x / (float)camera.width , 0.0f, 1.0f);
						float sy = std::clamp((float)y / (float)camera.height, 0.0f, 1.0f);
						printf("%f %f\n",sx,sy);
						if (sx < 0.5f) {
							camera.processPressKeyLeft(0.5f -  sx);
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
					if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
						camera.processPressKeyW(1.0f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
						camera.processPressKeyS(1.0f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
						camera.processPressKeyA(1.0f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
						camera.processPressKeyD(1.0f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
						camera.processPressKeyUp(0.5f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
						camera.processPressKeyDown(0.5f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
						camera.processPressKeyLeft(0.5f);
						update = true;
					}
					if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
						camera.processPressKeyRight(0.5f);
						update = true;
					}
					if (update) {
						auto [dir_u, dir_v, dir_w] = camera.getUVW();
						owlRayGenSet3fv(raygen, "camera.eye"  , (const float*)&camera.origin);
						owlRayGenSet3fv(raygen, "camera.dir_u", (const float*)&dir_u);
						owlRayGenSet3fv(raygen, "camera.dir_v", (const float*)&dir_v);
						owlRayGenSet3fv(raygen, "camera.dir_w", (const float*)&dir_w);
					}
				}
			}
			
			owlRayGenSetPointer(raygen, "fb_data", viewer->mapFramePtr());
			owlBuildSBT(context, OWL_SBT_RAYGENS);
			owlLaunch2D(raygen, camera.width, camera.height, params);
			viewer->unmapFramePtr();
			viewer->render();
			glfwSwapBuffers(window);
		}
		viewer.reset();
		glfwHideWindow(window);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	return 0;
}