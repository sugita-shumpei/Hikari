find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
add_library(hikari-core-ui STATIC)
target_sources(hikari-core-ui PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/ui/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/ui/app.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/ui/manager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/ui/renderer.h
  
  ${CMAKE_CURRENT_SOURCE_DIR}/src/app.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.cpp
)
target_include_directories(hikari-core-ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PUBLIC ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})
target_link_libraries(hikari-core-ui PUBLIC hikari-core-window imgui::imgui)
target_compile_features(hikari-core-ui PRIVATE cxx_std_20)
target_compile_options(hikari-core-ui PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::ui ALIAS hikari-core-ui)
