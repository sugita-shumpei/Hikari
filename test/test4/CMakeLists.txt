configure_file(test4_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test4_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

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
 hikari/core/node.h   "hikari/core/resource.h" "hikari/core/resource.cpp" "hikari/core/blob.h" "hikari/core/blob.cpp" "hikari/core/utils.h" "hikari/core/utils.cpp" "hikari/core/serializer.h" "hikari/core/serializer.cpp")
target_compile_features(hikari-test-test4 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test4 PRIVATE /utf-8)
target_include_directories(hikari-test-test4
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
)
target_link_libraries(hikari-test-test4 PRIVATE
  glm::glm
  tinyxml2::tinyxml2
  scn::scn
  fmt::fmt
  ZLIB::ZLIB tinyply
  tinyobjloader::tinyobjloader
  nlohmann_json::nlohmann_json
)
