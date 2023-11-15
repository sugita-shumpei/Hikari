add_library(hikari SHARED)

target_compile_features(hikari PRIVATE cxx_std_17)
target_compile_options(hikari PRIVATE "/Zc:__cplusplus")
target_sources(hikari PRIVATE 
data_type.h
platform.h
uuid.h
object.h 
object.cpp 
ref_ptr.h
ref_cnt_object.h
ref_cnt_object.cpp
transform_graph.h
transform_graph.cpp
shape.h
shape.cpp
math/common.h
math/common.cpp
math/affine_3d.h
math/matrix.h
math/quat.h
math/vec.h
math/aabb.h
shape/circle.h
shape/circle.cpp
shape/sphere.h
shape/sphere.cpp
shape/mesh.h
shape/mesh.cpp
shape/quad.h
shape/quad.cpp
shape/cube.h
shape/cube.cpp
shape/cylinder.h
shape/cylinder.cpp
)

target_include_directories(hikari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(hikari::hikari ALIAS hikari)
