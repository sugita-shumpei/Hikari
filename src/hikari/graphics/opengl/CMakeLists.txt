find_package(glad CONFIG REQUIRED)

add_library(hikari-graphics-opengl SHARED)
target_compile_features(hikari-graphics-opengl PRIVATE cxx_std_17)
target_compile_options(hikari-graphics-opengl  PRIVATE "/Zc:__cplusplus")
target_sources(hikari-graphics-opengl PRIVATE 
interface/hikari/graphics/opengl/context.h
interface/hikari/graphics/opengl/entry.h
interface/hikari/graphics/opengl/plugin.h
src/context.cpp
src/entry.cpp
src/plugin.cpp
)

target_include_directories(hikari-graphics-opengl 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../common/interface
${CMAKE_CURRENT_SOURCE_DIR}/../../core/interface
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(hikari-graphics-opengl PRIVATE hikari::graphics::common hikari::core glad::glad)
add_library(hikari::graphics::opengl ALIAS hikari-graphics-opengl)

add_library(hikari-graphics-opengl-headers INTERFACE)
target_include_directories(hikari-graphics-opengl-headers 
INTERFACE 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../common/include
${CMAKE_CURRENT_SOURCE_DIR}/../../core/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../../math/interface 
)
add_library(hikari::graphics::opengl::headers ALIAS hikari-graphics-opengl-headers)

set_target_properties(hikari-graphics-opengl         PROPERTIES FOLDER "src/hikari/graphics/opengl")
set_target_properties(hikari-graphics-opengl-headers PROPERTIES FOLDER "src/hikari/graphics/opengl")

