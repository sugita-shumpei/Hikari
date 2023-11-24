add_library(hikari-shape SHARED)
find_package(tinyobjloader CONFIG REQUIRED)
target_compile_features(hikari-shape PRIVATE cxx_std_17)
target_compile_options(hikari-shape PRIVATE "/Zc:__cplusplus")
target_sources(hikari-shape PRIVATE 

interface/hikari/shape/circle.h
interface/hikari/shape/sphere.h
interface/hikari/shape/mesh.h
interface/hikari/shape/obj_mesh.h
interface/hikari/shape/quad.h
interface/hikari/shape/cube.h
interface/hikari/shape/cylinder.h
interface/hikari/shape/plugin.h

src/circle.cpp
src/sphere.cpp
src/mesh.cpp
src/obj_mesh.cpp
src/quad.cpp
src/cube.cpp
src/cylinder.cpp
src/plugin.cpp
)

target_include_directories(hikari-shape 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../core/include
)

target_link_libraries(hikari-shape PRIVATE hikari::core tinyobjloader::tinyobjloader)
add_library(hikari::shape ALIAS hikari-shape)

add_library(hikari-shape-headers INTERFACE)
target_include_directories(hikari-shape-headers 
INTERFACE 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../core/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../math/interface 
)
add_library(hikari::shape::headers ALIAS hikari-shape-headers)

set_target_properties(hikari-shape         PROPERTIES FOLDER "src/hikari/shape")
set_target_properties(hikari-shape-headers PROPERTIES FOLDER "src/hikari/shape")

