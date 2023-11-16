#include <iostream>
#include <string>
#include <owl/owl.h>
#include <GLFW/glfw3.h>
#include <gl_viewer.h>
#include <vector>
#include "hostCode.h"
#include "deviceCode.h"

extern "C" char* deviceCode_ptx[];

int main() {
	int width = 1024; int height = 1024;

	auto context      = owlContextCreate();
	auto module       = owlModuleCreate(context, (const char*)deviceCode_ptx);
	auto accum_buffer = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT4, width * height, nullptr);
	auto params       = static_cast<OWLParams>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP ,offsetof(LaunchParams,world)       },
			OWLVarDecl{"accum_buffer",OWLDataType::OWL_BUFPTR,offsetof(LaunchParams,accum_buffer)},
			OWLVarDecl{"accum_sample",OWLDataType::OWL_INT   ,offsetof(LaunchParams,accum_sample)},
			OWLVarDecl{nullptr}
		};
		params = owlParamsCreate(context, sizeof(LaunchParams), var_decls, -1);
		owlParamsSetBuffer(params, "accum_buffer", accum_buffer);
		owlParamsSet1i(params, "accum_sample", 0);
	}

	auto raygen     = static_cast<OWLRayGen>(nullptr);
	{
		OWLVarDecl var_decls[] = {
			OWLVarDecl{"world"       ,OWLDataType::OWL_GROUP      ,offsetof(LaunchParams,world)},
			OWLVarDecl{"fb_data"     ,OWLDataType::OWL_RAW_POINTER,offsetof(RayGenData,fb_data)},
			OWLVarDecl{"fb_size"     ,OWLDataType::OWL_INT2       ,offsetof(RayGenData,fb_size)},
			OWLVarDecl{nullptr}
		};
		raygen = owlRayGenCreate(context, module, "simpleRG", sizeof(RayGenData), var_decls, -1);
		owlRayGenSetPointer(raygen, "fb_data", nullptr);
		owlRayGenSet2i(raygen, "fb_size"     , width, height);
	}

	auto miss_prog  = static_cast<OWLMissProg>(nullptr);
	{
		miss_prog = owlMissProgCreate(context, module, "simpleMS", sizeof(MissProgData), nullptr, 0);
	}

	auto geom_type  = static_cast<OWLGeomType>(nullptr);
	{
		geom_type  = owlGeomTypeCreate(context, OWLGeomKind::OWL_GEOM_TRIANGLES, sizeof(HitgroupData), nullptr, 0);
		owlGeomTypeSetClosestHit(geom_type, 0, module, "simpleCH");
	}

	auto trim       = static_cast<OWLGeom>(nullptr);
	{
		auto vert_buf = static_cast<OWLBuffer>(nullptr);
		auto indx_buf = static_cast<OWLBuffer>(nullptr);
		{
			const std::vector<owl::vec3f> vertices = {
				owl::vec3f(-1.0f,-1.0f,1.0f),
				owl::vec3f( 1.0f,-1.0f,1.0f),
				owl::vec3f( 0.0f, 1.0f,1.0f)
			};

			const std::vector<owl::vec3i> indices = {
				owl::vec3i(0,1,2),
			};

			vert_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_FLOAT3, vertices.size(), vertices.data());
			indx_buf = owlDeviceBufferCreate(context, OWLDataType::OWL_INT3  ,  indices.size(),  indices.data());
		}
		trim = owlGeomCreate(context, geom_type);
		owlTrianglesSetVertices(trim, vert_buf, owlBufferSizeInBytes(vert_buf) / sizeof(owl::vec3f), sizeof(owl::vec3f), 0);
		owlTrianglesSetIndices (trim, indx_buf, owlBufferSizeInBytes(indx_buf) / sizeof(owl::vec3i), sizeof(owl::vec3i), 0);
	}

	auto trim_group = static_cast<OWLGroup>(nullptr);
	{
		trim_group  = owlTrianglesGeomGroupCreate(context, 1, &trim, 0);
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
		GLFWwindow* window = glfwCreateWindow(width, height, "title", nullptr, nullptr);
		glfwMakeContextCurrent(window);
		if (!hikari::test::owl::testlib::loadGLLoader((hikari::test::owl::testlib::GLloadproc)glfwGetProcAddress)){
			return -1;
		}
		auto viewer = std::make_unique<hikari::test::owl::testlib::GLViewer>(context, width, height);
		glfwShowWindow(window);
		while (!glfwWindowShouldClose(window)) {
			glfwGetWindowSize(window, &width, &height);
			if (viewer->resize(width, height)) {
				printf("%d %d\n", width, height);
				owlBufferResize(accum_buffer, width * height);
				owlParamsSetBuffer(params, "accum_buffer", accum_buffer);
				owlParamsSet1i(    params, "accum_sample", 0);
				owlRayGenSet2i(    raygen, "fb_size"     , width,height);
			}
			owlRayGenSetPointer(raygen, "fb_data", viewer->mapFramePtr());
			owlBuildSBT(context, OWL_SBT_RAYGENS);
			owlLaunch2D(raygen, width, height, params);
			viewer->unmapFramePtr();
			viewer->render();
			glfwPollEvents();
			glfwSwapBuffers(window);
		}
		viewer.reset();
		glfwHideWindow(window);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	return 0;
}