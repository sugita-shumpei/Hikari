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
hikari/core/types/aabb.cpp
hikari/core/types/aabb.h
hikari/core/types/color.cpp
hikari/core/types/color.h
hikari/core/types/data_type.cpp
hikari/core/types/data_type.h
hikari/core/types/matrix.cpp
hikari/core/types/matrix.h
hikari/core/types/object.cpp
hikari/core/types/object.h
hikari/core/types/properties.cpp
hikari/core/types/properties.h
hikari/core/types/property.cpp
hikari/core/types/property.h
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
