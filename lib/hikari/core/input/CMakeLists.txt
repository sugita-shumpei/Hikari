add_library(hikari-core-input    STATIC)
target_sources(hikari-core-input PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/input/common.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/input/event.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/input/input.h
${CMAKE_CURRENT_SOURCE_DIR}/src/event.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/input.cpp
)
target_include_directories(hikari-core-input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hikari-core-input PUBLIC hikari-core-common hikari-core-event)
target_compile_features(hikari-core-input PRIVATE cxx_std_20)
target_compile_options(hikari-core-input PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::input ALIAS hikari-core-input)
