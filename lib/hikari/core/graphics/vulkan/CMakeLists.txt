find_package(Vulkan REQUIRED)
find_package(function2 CONFIG REQUIRED)
add_library(hikari-core-graphics-vulkan    STATIC)
target_sources(hikari-core-graphics-vulkan PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/common.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/context.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/device.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/instance.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/swapchain.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/renderer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/graphics/vulkan/ui_renderer.h
${CMAKE_CURRENT_SOURCE_DIR}/src/instance.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/device.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/swapchain.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/context.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/ui_renderer.cpp
)
target_include_directories(hikari-core-graphics-vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hikari-core-graphics-vulkan PUBLIC hikari-core-window Vulkan::Headers function2::function2)
target_compile_features(hikari-core-graphics-vulkan PRIVATE cxx_std_20)
target_compile_options(hikari-core-graphics-vulkan PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::graphics::vulkan ALIAS hikari-core-graphics-vulkan)

add_executable(hikari-core-graphics-vulkan-test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp)
target_compile_features(hikari-core-graphics-vulkan-test PRIVATE cxx_std_20)
target_compile_options(hikari-core-graphics-vulkan-test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(hikari-core-graphics-vulkan-test PRIVATE hikari::core::graphics::vulkan)
