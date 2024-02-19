configure_file(test4_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test4_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Imath         CONFIG REQUIRED)
find_package(Matplot++     CONFIG REQUIRED)
find_package(Vulkan               REQUIRED)
find_package(glfw3         CONFIG REQUIRED)
find_package(glad          CONFIG REQUIRED)
find_package(spdlog        CONFIG REQUIRED)
find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS BS_thread_pool.hpp)
# find_package(Python COMPONENTS Interpreter Development NumPy)
# find_package(pybind11 CONFIG)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS  matplotlibcpp.h )
add_executable(hikari-test-test4
 main.cpp
 hikari/core/common.h
 hikari/core/object.h
 hikari/core/object.cpp
 hikari/core/data_type.h
 hikari/core/property.h
 hikari/core/property.cpp
 hikari/core/tuple.h
 hikari/core/tuple.cpp
 hikari/core/transform.h
 hikari/core/transform.cpp
 hikari/core/field.h
 hikari/core/field.cpp
 hikari/core/node.cpp
 hikari/core/node.h
 hikari/core/resource.h
 hikari/core/resource.cpp
 hikari/core/blob.h
 hikari/core/blob.cpp
 hikari/core/utils.h
 hikari/core/utils.cpp
 hikari/core/serializer.h
 hikari/core/serializer.cpp
 hikari/core/deserializer.cpp
 hikari/core/deserializer.h
 hikari/core/shape.h 
 hikari/core/shape.cpp 
 hikari/core/light.h 
 hikari/core/light.cpp 
 hikari/core/camera.h 
 hikari/core/camera.cpp 
 hikari/core/material.h 
 hikari/core/material.cpp 
 hikari/core/texture.h 
 hikari/core/texture.cpp 
 hikari/core/spectrum.h 
 hikari/core/spectrum.cpp 
 hikari/core/color.h 
 hikari/core/color.cpp 
 hikari/core/bbox.h
 hikari/core/bbox.cpp
 hikari/spectrum/color.h 
 hikari/spectrum/color.cpp 
 hikari/spectrum/regular.h 
 hikari/spectrum/regular.cpp 
 hikari/spectrum/blackbody.h 
 hikari/spectrum/blackbody.cpp 
 hikari/spectrum/uniform.h 
 hikari/spectrum/uniform.cpp 
 hikari/spectrum/irregular.h 
 hikari/spectrum/irregular.cpp 
 hikari/texture/mipmap.h 
 hikari/texture/mipmap.cpp 
 hikari/texture/checker.h 
 hikari/texture/checker.cpp 
 hikari/shape/mesh.h 
 hikari/shape/mesh.cpp 
 hikari/shape/cube.h 
 hikari/shape/cube.cpp 
 hikari/shape/rect.h
 hikari/shape/rect.cpp 
 hikari/shape/sphere.h 
 hikari/shape/sphere.cpp
 hikari/bsdf/delta_conductor.h
 hikari/bsdf/delta_conductor.cpp
 hikari/bsdf/rough_conductor.h
 hikari/bsdf/rough_conductor.cpp
 hikari/bsdf/delta_plastic.h
 hikari/bsdf/delta_plastic.cpp
 hikari/bsdf/rough_plastic.h
 hikari/bsdf/rough_plastic.cpp
 hikari/bsdf/delta_dielectric.h
 hikari/bsdf/delta_dielectric.cpp
 hikari/bsdf/rough_dielectric.cpp
 hikari/bsdf/rough_dielectric.h
 hikari/bsdf/thin_dielectric.h
 hikari/bsdf/thin_dielectric.cpp
 hikari/bsdf/diffuse.h
 hikari/bsdf/diffuse.cpp
 hikari/bsdf/blend.h
 hikari/bsdf/blend.cpp
 hikari/light/area.h
 hikari/light/area.cpp
 hikari/light/constant.h
 hikari/light/constant.cpp
 hikari/light/directional.h
 hikari/light/directional.cpp
 hikari/light/directional_area.h
 hikari/light/directional_area.cpp
 hikari/light/envmap.h
 hikari/light/envmap.cpp
 hikari/light/point.h
 hikari/light/point.cpp
 hikari/light/projector.h
 hikari/light/projector.cpp
 hikari/light/spot.h
 hikari/light/spot.cpp
 hikari/render/gfx_window.cpp
 hikari/render/gfx_instance.h
 hikari/render/gfx_instance.cpp
 hikari/render/gfx_common.h
 hikari/render/gfx_common.cpp
 hikari/render/gfx_device.h
 hikari/render/gfx_device.cpp
 hikari/platform/glfw/glfw_window_manager.h
 hikari/platform/glfw/glfw_window_manager.cpp
 hikari/platform/glfw/render/gfx_glfw_window.h
 hikari/platform/glfw/render/gfx_glfw_window.cpp
 hikari/platform/common/render/gfx_common_instance.h
 hikari/platform/common/render/gfx_common_instance.cpp
 hikari/platform/opengl/render/gfx_opengl_instance.h
 hikari/platform/opengl/render/gfx_opengl_instance.cpp
 hikari/platform/opengl/render/gfx_opengl_window.h
 hikari/platform/opengl/render/gfx_opengl_window.cpp
 hikari/platform/opengl/render/gfx_opengl_device.h
 hikari/platform/opengl/render/gfx_opengl_device.cpp
 hikari/platform/vulkan/render/gfx_vulkan_common.h
 hikari/platform/vulkan/render/gfx_vulkan_window.h
 hikari/platform/vulkan/render/gfx_vulkan_window.cpp
 hikari/platform/vulkan/render/gfx_vulkan_instance.h
 hikari/platform/vulkan/render/gfx_vulkan_instance.cpp
)
target_compile_features(hikari-test-test4 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test4 PRIVATE /utf-8)
target_include_directories(hikari-test-test4
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
  # ${MATPLOTLIB_CPP_INCLUDE_DIRS}
  # ${Python2_INCLUDE_DIRS}
  ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS}
)
target_link_libraries(hikari-test-test4 PRIVATE
  glad::glad
  glm::glm
  tinyxml2::tinyxml2
  scn::scn
  fmt::fmt
  ZLIB::ZLIB
  tinyply
  tinyobjloader::tinyobjloader
  nlohmann_json::nlohmann_json
  Imath::Imath
  Matplot++::cimg
  Matplot++::matplot
  Vulkan::Vulkan
  spdlog::spdlog
  glfw
#  Python::Python
#  Python::NumPy
)

add_custom_command(TARGET hikari-test-test4 POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:hikari-test-test4> $<TARGET_FILE_DIR:hikari-test-test4>
COMMAND_EXPAND_LISTS
)
