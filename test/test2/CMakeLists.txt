configure_file(test2_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test2_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(hikari-test-test2
main.cpp
)
target_compile_features(hikari-test-test2 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test2 PRIVATE /utf-8)
target_include_directories(hikari-test-test2
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
)
target_link_libraries(hikari-test-test2 PRIVATE
  glm::glm
  tinyxml2::tinyxml2
  scn::scn
  fmt::fmt
  ZLIB::ZLIB tinyply
  tinyobjloader::tinyobjloader
  nlohmann_json::nlohmann_json
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/lib/slang.lib
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/lib/gfx.lib
)
add_custom_command(TARGET hikari-test-test2 POST_BUILD  
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang.dll         $<TARGET_FILE_DIR:hikari-test-test2>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang-llvm.dll    $<TARGET_FILE_DIR:hikari-test-test2>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/slang-glslang.dll $<TARGET_FILE_DIR:hikari-test-test2>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin/gfx.dll           $<TARGET_FILE_DIR:hikari-test-test2>
)
