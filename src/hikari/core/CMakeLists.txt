find_package(glm     CONFIG REQUIRED)
find_package(Imath   CONFIG REQUIRED)
find_package(stduuid CONFIG REQUIRED)
add_library(hikari-core STATIC)
target_compile_features(hikari-core PRIVATE cxx_std_20)
target_compile_options(hikari-core PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-core PUBLIC include)
target_sources(hikari-core PRIVATE
include/hikari/core/data_type.h
include/hikari/core/bitmap.h
include/hikari/core/bsdf.h
include/hikari/core/blob.h
include/hikari/core/camera.h
include/hikari/core/color.h
include/hikari/core/film.h
include/hikari/core/light.h
include/hikari/core/material.h
include/hikari/core/medium.h
include/hikari/core/mipmap.h
include/hikari/core/node.h
include/hikari/core/scene.h
include/hikari/core/spectrum.h
include/hikari/core/variant.h
include/hikari/core/shape.h
include/hikari/core/texture.h
include/hikari/core/transform.h
include/hikari/core/utils.h

src/blob.cpp
src/bitmap.cpp
src/bsdf.cpp
src/camera.cpp
src/color.cpp
src/film.cpp
src/light.cpp
src/material.cpp
src/medium.cpp
src/mipmap.cpp
src/node.cpp
src/scene.cpp
src/variant.cpp
src/spectrum.cpp
src/shape.cpp
src/texture.cpp
)
target_link_libraries(hikari-core PUBLIC glm::glm Imath::Imath stduuid)

add_executable(hikari-core-test src/test.cpp)
target_compile_features(hikari-core-test PRIVATE cxx_std_20)
target_compile_options(hikari-core-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-core-test   PRIVATE hikari-core )

set_target_properties(hikari-core      PROPERTIES FOLDER "src/hikari/core")
set_target_properties(hikari-core-test PROPERTIES FOLDER "src/hikari/core/test")
