find_package(Vulkan REQUIRED)
find_package(function2 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
add_library(hikari-core-graphics-opengl    STATIC)
target_sources(hikari-core-graphics-opengl PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/opengl/context.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/opengl/renderer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/opengl/ui_renderer.h
${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/ui_renderer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/context.cpp
)
target_include_directories(hikari-core-graphics-opengl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PUBLIC ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})
target_link_libraries(hikari-core-graphics-opengl PUBLIC hikari-core-window hikari-core-ui Vulkan::Headers function2::function2 glad::glad)
target_compile_features(hikari-core-graphics-opengl PRIVATE cxx_std_20)
target_compile_options(hikari-core-graphics-opengl PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::graphics::opengl ALIAS hikari-core-graphics-opengl)

add_executable(hikari-core-graphics-opengl-test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_compile_features(hikari-core-graphics-opengl-test PRIVATE cxx_std_20)
target_compile_options(hikari-core-graphics-opengl-test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(hikari-core-graphics-opengl-test PRIVATE hikari::core::graphics::opengl)
