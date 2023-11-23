add_library(hikari-headers INTERFACE)
target_include_directories(hikari-headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/interface)
add_library(hikari::headers ALIAS hikari-headers)

add_library(hikari SHARED)
target_compile_features(hikari PRIVATE cxx_std_17)
target_compile_options(hikari PRIVATE "/Zc:__cplusplus")
target_compile_definitions(hikari PRIVATE -DHK_MATH_USE_STD_CXX)
target_sources(hikari PRIVATE 

interface/hikari/data_type.h
interface/hikari/platform.h
interface/hikari/uuid.h
interface/hikari/object.h 
interface/hikari/object_array.h 
interface/hikari/value_array.h
interface/hikari/ref_ptr.h
interface/hikari/dynamic_loader.h
interface/hikari/plugin.h
interface/hikari/shape.h

include/hikari/ref_cnt_object.h

src/hikari/object.cpp 
src/hikari/object_array.cpp 
src/hikari/value_array.cpp
src/hikari/ref_cnt_object.cpp
src/hikari/plugin.cpp
src/hikari/shape.cpp
)
target_include_directories(hikari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(hikari::hikari ALIAS hikari)
set_target_properties(hikari PROPERTIES FOLDER "src/hikari")


add_library(hikari-math SHARED)
target_compile_features(hikari-math PRIVATE cxx_std_17)
target_compile_options(hikari-math  PRIVATE "/Zc:__cplusplus")
target_sources(hikari-math PRIVATE 

interface/hikari/math/common.h
interface/hikari/math/affine_3d.h
interface/hikari/math/matrix.h
interface/hikari/math/quat.h
interface/hikari/math/vec.h
interface/hikari/math/aabb.h

src/hikari/math/common.cpp
)
target_include_directories(hikari-math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(hikari::math ALIAS hikari)
set_target_properties(hikari-math PROPERTIES FOLDER "src/hikari/math")

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

src/hikari/shape/circle.cpp
src/hikari/shape/sphere.cpp
src/hikari/shape/mesh.cpp
src/hikari/shape/obj_mesh.cpp
src/hikari/shape/quad.cpp
src/hikari/shape/cube.cpp
src/hikari/shape/cylinder.cpp
src/hikari/shape/plugin.cpp
)

target_include_directories(hikari-shape PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hikari-shape PRIVATE hikari tinyobjloader::tinyobjloader)
add_library(hikari::shape ALIAS hikari-shape)
set_target_properties(hikari-shape PROPERTIES FOLDER "src/hikari/shape")
