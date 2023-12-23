find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-spectrum STATIC)
target_compile_features(hikari-spectrum PRIVATE cxx_std_20)
target_compile_options(hikari-spectrum PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-spectrum PUBLIC include)
target_sources(hikari-spectrum PRIVATE
    include/hikari/spectrum/srgb.h
    include/hikari/spectrum/uniform.h
    src/srgb.cpp
    src/uniform.cpp
)
target_link_libraries(hikari-spectrum PUBLIC hikari-core)

add_executable(hikari-spectrum-test src/test.cpp)
target_compile_features(hikari-spectrum-test PRIVATE cxx_std_20)
target_compile_options(hikari-spectrum-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-spectrum-test   PRIVATE hikari-spectrum)

set_target_properties(hikari-spectrum      PROPERTIES FOLDER "src/hikari/spectrum")
set_target_properties(hikari-spectrum-test PROPERTIES FOLDER "src/hikari/spectrum/test")
