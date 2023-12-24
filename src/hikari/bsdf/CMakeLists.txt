find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-bsdf STATIC)
target_compile_features(hikari-bsdf PRIVATE cxx_std_20)
target_compile_options(hikari-bsdf PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-bsdf PUBLIC include)
target_sources(hikari-bsdf PRIVATE
    include/hikari/bsdf/bump_map.h
    include/hikari/bsdf/conductor.h
    include/hikari/bsdf/dielectric.h
    include/hikari/bsdf/diffuse.h
    include/hikari/bsdf/mask.h
    include/hikari/bsdf/normal_map.h
    include/hikari/bsdf/phong.h
    include/hikari/bsdf/plastic.h
    include/hikari/bsdf/rough_conductor.h
    include/hikari/bsdf/rough_dielectric.h
    include/hikari/bsdf/rough_plastic.h
    include/hikari/bsdf/thin_dielectric.h
    include/hikari/bsdf/two_sided.h
    src/bump_map.cpp
    src/conductor.cpp
    src/dielectric.cpp
    src/diffuse.cpp
    src/mask.cpp
    src/normal_map.cpp
    src/phong.cpp
    src/plastic.cpp
    src/rough_conductor.cpp
    src/rough_dielectric.cpp
    src/rough_plastic.cpp
    src/thin_dielectric.cpp
    src/two_sided.cpp
)
target_link_libraries(hikari-bsdf PUBLIC hikari-core hikari-spectrum hikari-texture)

add_executable(hikari-bsdf-test src/test.cpp)
target_compile_features(hikari-bsdf-test PRIVATE cxx_std_20)
target_compile_options(hikari-bsdf-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-bsdf-test   PRIVATE hikari-bsdf)

set_target_properties(hikari-bsdf      PROPERTIES FOLDER "src/hikari/bsdf")
set_target_properties(hikari-bsdf-test PROPERTIES FOLDER "src/hikari/bsdf/test")
