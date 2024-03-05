add_library(hikari-core-app    STATIC)
target_sources(hikari-core-app PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/app/app.h
${CMAKE_CURRENT_SOURCE_DIR}/src/app.cpp
)
target_include_directories(hikari-core-app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hikari-core-app PRIVATE cxx_std_20)
target_compile_options(hikari-core-app PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::app ALIAS hikari-core-app)
