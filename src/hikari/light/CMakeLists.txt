find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-light STATIC)
target_compile_features(hikari-light PRIVATE cxx_std_20)
target_compile_options(hikari-light PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-light PUBLIC include)
target_sources(hikari-light PRIVATE
    include/hikari/light/area.h
    include/hikari/light/constant.h
    include/hikari/light/envmap.h
    include/hikari/light/directional.h
    src/area.cpp
    src/constant.cpp
    src/envmap.cpp
    src/directional.cpp
)
target_link_libraries(hikari-light PUBLIC hikari-core)

add_executable(hikari-light-test src/test.cpp)
target_compile_features(hikari-light-test PRIVATE cxx_std_20)
target_compile_options(hikari-light-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-light-test   PRIVATE hikari-light)

set_target_properties(hikari-light      PROPERTIES FOLDER "src/hikari/light")
set_target_properties(hikari-light-test PROPERTIES FOLDER "src/hikari/light/test")
