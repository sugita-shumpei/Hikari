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

src/common.cpp
)

target_include_directories(hikari-math 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../core/interface 
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../core/include
)
add_library(hikari::math ALIAS hikari-math)

add_library(hikari-math-headers INTERFACE)
target_include_directories(hikari-math-headers 
INTERFACE 
${CMAKE_CURRENT_SOURCE_DIR}/interface
${CMAKE_CURRENT_SOURCE_DIR}/../core/interface 
)
add_library(hikari::math::headers ALIAS hikari-math-headers)

set_target_properties(hikari-math         PROPERTIES FOLDER "src/hikari/math")
set_target_properties(hikari-math-headers PROPERTIES FOLDER "src/hikari/math")