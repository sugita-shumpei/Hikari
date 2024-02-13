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
find_package(Matplot++ CONFIG REQUIRED)
# find_package(Python COMPONENTS Interpreter Development NumPy)
# find_package(pybind11 CONFIG)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS  matplotlibcpp.h )
add_executable(hikari-test-test4
 main.cpp
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
 hikari/shape/mesh.h 
 hikari/shape/mesh.cpp 
 hikari/shape/cube.h 
 hikari/shape/cube.cpp 
 hikari/shape/quad.h 
 hikari/shape/quad.cpp 
 hikari/shape/sphere.h 
 hikari/shape/sphere.cpp 
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
)
target_link_libraries(hikari-test-test4 PRIVATE
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
#  Python::Python
#  Python::NumPy
)

add_custom_command(TARGET hikari-test-test4 POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:hikari-test-test4> $<TARGET_FILE_DIR:hikari-test-test4>
COMMAND_EXPAND_LISTS
)
