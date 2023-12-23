find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-camera STATIC)
target_compile_features(hikari-camera PRIVATE cxx_std_20)
target_compile_options(hikari-camera PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-camera PUBLIC include)
target_sources(hikari-camera PRIVATE
    include/hikari/camera/orthographic.h
    include/hikari/camera/perspective.h
    src/orthographic.cpp
    src/perspective.cpp
)
target_link_libraries(hikari-camera PUBLIC hikari-core)

add_executable(hikari-camera-test src/test.cpp)
target_compile_features(hikari-camera-test PRIVATE cxx_std_20)
target_compile_options(hikari-camera-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-camera-test   PRIVATE hikari-camera)

set_target_properties(hikari-camera      PROPERTIES FOLDER "src/hikari/camera")
set_target_properties(hikari-camera-test PROPERTIES FOLDER "src/hikari/camera/test")
