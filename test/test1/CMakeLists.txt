configure_file(test1_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test1_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(hikari-test-test1
main.cpp
)
target_compile_features(hikari-test-test1 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test1 PRIVATE /utf-8)
target_include_directories(hikari-test-test1
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
)
target_link_libraries(hikari-test-test1 PRIVATE
  glm::glm
  tinyxml2::tinyxml2
  scn::scn
  fmt::fmt
  ZLIB::ZLIB tinyply
  tinyobjloader::tinyobjloader
  nlohmann_json::nlohmann_json
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/lib/slang.lib
)
add_custom_command(TARGET hikari-test-test1 POST_BUILD  
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang.dll         $<TARGET_FILE_DIR:hikari-test-test1>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang-llvm.dll    $<TARGET_FILE_DIR:hikari-test-test1>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang-glslang.dll $<TARGET_FILE_DIR:hikari-test-test1>
)
