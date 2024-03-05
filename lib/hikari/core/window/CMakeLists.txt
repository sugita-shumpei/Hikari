find_package(glfw3 CONFIG REQUIRED)
add_library(hikari-core-window STATIC)
target_sources(hikari-core-window PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/window/app.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/window/common.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/window/event.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/window/renderer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/window/window.h
${CMAKE_CURRENT_SOURCE_DIR}/src/app.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/event.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/window.cpp
)
target_include_directories(hikari-core-window PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PUBLIC ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})
target_link_libraries(hikari-core-window PUBLIC hikari-core-app hikari-core-common hikari-core-event hikari-core-input PRIVATE glfw)
target_compile_features(hikari-core-window PRIVATE cxx_std_20)
target_compile_options(hikari-core-window PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::window ALIAS hikari-core-window)
