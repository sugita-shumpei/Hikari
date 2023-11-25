add_library(hikari-graphics-common SHARED)
target_compile_features(hikari-graphics-common PRIVATE cxx_std_17)
target_compile_options(hikari-graphics-common PRIVATE "/Zc:__cplusplus")
target_sources(hikari-graphics-common PRIVATE 
interface/hikari/graphics/entry.h
interface/hikari/graphics/instance.h
interface/hikari/graphics/plugin.h
src/entry.cpp
src/instance.cpp
src/plugin.cpp
)

target_include_directories(hikari-graphics-common 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(hikari-graphics-common PRIVATE hikari::core )
add_library(hikari::graphics::common ALIAS hikari-graphics-common)

add_library(hikari-graphics-common-headers INTERFACE)
target_include_directories(hikari-graphics-common-headers 
INTERFACE 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../../core/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../../math/interface 
)
add_library(hikari::graphics::common::headers ALIAS hikari-graphics-common-headers)

set_target_properties(hikari-graphics-common         PROPERTIES FOLDER "src/hikari/graphics/common")
set_target_properties(hikari-graphics-common-headers PROPERTIES FOLDER "src/hikari/graphics/common")

