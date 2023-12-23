find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-shape STATIC)
target_compile_features(hikari-shape PRIVATE cxx_std_20)
target_compile_options(hikari-shape PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-shape PUBLIC include)
target_sources(hikari-shape PRIVATE
    include/hikari/shape/mesh.h
    include/hikari/shape/sphere.h
    include/hikari/shape/triangle.h
    include/hikari/shape/rectangle.h
    include/hikari/shape/cube.h
    src/mesh.cpp
    src/sphere.cpp
    src/triangle.cpp
    src/rectangle.cpp
    src/cube.cpp
)
target_link_libraries(hikari-shape PUBLIC hikari-core)

add_executable(hikari-shape-test src/test.cpp)
target_compile_features(hikari-shape-test PRIVATE cxx_std_20)
target_compile_options(hikari-shape-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-shape-test   PRIVATE hikari-shape)

set_target_properties(hikari-shape      PROPERTIES FOLDER "src/hikari/shape")
set_target_properties(hikari-shape-test PROPERTIES FOLDER "src/hikari/shape/test")