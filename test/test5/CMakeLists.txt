configure_file(test5_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test5_config.h)
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
add_executable(hikari-test-test5
 main.cpp
 hikari/framework/core/common.h
 hikari/framework/core/object.h
 hikari/framework/core/object.cpp
 hikari/framework/core/data_type.h
 hikari/framework/core/property.h
 hikari/framework/core/property.cpp
 hikari/framework/core/tuple.h
 hikari/framework/core/tuple.cpp
 hikari/framework/core/transform.h
 hikari/framework/core/transform.cpp
 hikari/framework/core/blob.h
 hikari/framework/core/blob.cpp
 hikari/framework/core/utils.h
 hikari/framework/core/utils.cpp
 hikari/framework/core/serializer.h
 hikari/framework/core/serializer.cpp
 hikari/framework/core/deserializer.cpp
 hikari/framework/core/deserializer.h
 hikari/framework/core/bbox.h
 hikari/framework/core/bbox.cpp
#  hikari/framework/scene/common/node.cpp
#  hikari/framework/scene/common/node.h
#  hikari/framework/scene/common/shape.h 
#  hikari/framework/scene/common/shape.cpp 
#  hikari/framework/scene/common/light.h 
#  hikari/framework/scene/common/light.cpp 
#  hikari/framework/scene/common/camera.h 
#  hikari/framework/scene/common/camera.cpp 
#  hikari/framework/scene/common/material.h 
#  hikari/framework/scene/common/material.cpp 
#  hikari/framework/scene/common/texture.h 
#  hikari/framework/scene/common/texture.cpp 
#  hikari/framework/scene/common/spectrum.h 
#  hikari/framework/scene/common/spectrum.cpp 
#  hikari/framework/scene/common/color.h 
#  hikari/framework/scene/common/color.cpp 
#  hikari/framework/scene/spectrum/color.h 
#  hikari/framework/scene/spectrum/color.cpp 
#  hikari/framework/scene/spectrum/regular.h 
#  hikari/framework/scene/spectrum/regular.cpp 
#  hikari/framework/scene/spectrum/blackbody.h 
#  hikari/framework/scene/spectrum/blackbody.cpp 
#  hikari/framework/scene/spectrum/uniform.h 
#  hikari/framework/scene/spectrum/uniform.cpp 
#  hikari/framework/scene/spectrum/irregular.h 
#  hikari/framework/scene/spectrum/irregular.cpp 
#  hikari/framework/scene/texture/mipmap.h 
#  hikari/framework/scene/texture/mipmap.cpp 
#  hikari/framework/scene/texture/checker.h 
#  hikari/framework/scene/texture/checker.cpp 
#  hikari/framework/scene/shape/mesh.h 
#  hikari/framework/scene/shape/mesh.cpp 
#  hikari/framework/scene/shape/cube.h 
#  hikari/framework/scene/shape/cube.cpp 
#  hikari/framework/scene/shape/rect.h
#  hikari/framework/scene/shape/rect.cpp 
#  hikari/framework/scene/shape/sphere.h 
#  hikari/framework/scene/shape/sphere.cpp
#  hikari/framework/scene/bsdf/delta_conductor.h
#  hikari/framework/scene/bsdf/delta_conductor.cpp
#  hikari/framework/scene/bsdf/rough_conductor.h
#  hikari/framework/scene/bsdf/rough_conductor.cpp
#  hikari/framework/scene/bsdf/delta_plastic.h
#  hikari/framework/scene/bsdf/delta_plastic.cpp
#  hikari/framework/scene/bsdf/rough_plastic.h
#  hikari/framework/scene/bsdf/rough_plastic.cpp
#  hikari/framework/scene/bsdf/delta_dielectric.h
#  hikari/framework/scene/bsdf/delta_dielectric.cpp
#  hikari/framework/scene/bsdf/rough_dielectric.cpp
#  hikari/framework/scene/bsdf/rough_dielectric.h
#  hikari/framework/scene/bsdf/thin_dielectric.h
#  hikari/framework/scene/bsdf/thin_dielectric.cpp
#  hikari/framework/scene/bsdf/diffuse.h
#  hikari/framework/scene/bsdf/diffuse.cpp
#  hikari/framework/scene/bsdf/blend.h
#  hikari/framework/scene/bsdf/blend.cpp
#  hikari/framework/scene/light/area.h
#  hikari/framework/scene/light/area.cpp
#  hikari/framework/scene/light/constant.h
#  hikari/framework/scene/light/constant.cpp
#  hikari/framework/scene/light/directional.h
#  hikari/framework/scene/light/directional.cpp
#  hikari/framework/scene/light/directional_area.h
#  hikari/framework/scene/light/directional_area.cpp
#  hikari/framework/scene/light/envmap.h
#  hikari/framework/scene/light/envmap.cpp
#  hikari/framework/scene/light/point.h
#  hikari/framework/scene/light/point.cpp
#  hikari/framework/scene/light/projector.h
#  hikari/framework/scene/light/projector.cpp
#  hikari/framework/scene/light/spot.h
#  hikari/framework/scene/light/spot.cpp
#  hikari/framework/graphics/window.cpp
#  hikari/framework/graphics/instance.h
#  hikari/framework/graphics/instance.cpp
#  hikari/framework/graphics/common.h
#  hikari/framework/graphics/common.cpp
#  hikari/framework/graphics/device.h
#  hikari/framework/graphics/device.cpp
#  hikari/framework/graphics/opengl/instance.h
#  hikari/framework/graphics/opengl/instance.cpp
#  hikari/framework/graphics/opengl/window.h
#  hikari/framework/graphics/opengl/window.cpp
#  hikari/framework/graphics/opengl/device.h
#  hikari/framework/graphics/opengl/device.cpp
#  hikari/framework/graphics/vulkan/common.h
#  hikari/framework/graphics/vulkan/window.h
#  hikari/framework/graphics/vulkan/window.cpp
#  hikari/framework/graphics/vulkan/instance.h
#  hikari/framework/graphics/vulkan/instance.cpp
)
target_compile_features(hikari-test-test5 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test5 PRIVATE /utf-8)
target_include_directories(hikari-test-test5
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
  # ${MATPLOTLIB_CPP_INCLUDE_DIRS}
  # ${Python2_INCLUDE_DIRS}
  ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS}
)
target_link_libraries(hikari-test-test5 PRIVATE
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

add_custom_command(TARGET hikari-test-test5 POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:hikari-test-test5> $<TARGET_FILE_DIR:hikari-test-test5>
COMMAND_EXPAND_LISTS
)
