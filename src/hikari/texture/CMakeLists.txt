find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-texture STATIC)
target_compile_features(hikari-texture PRIVATE cxx_std_20)
target_compile_options(hikari-texture PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-texture PUBLIC include)
target_sources(hikari-texture PRIVATE
    include/hikari/texture/mipmap.h
    include/hikari/texture/checkerboard.h
    src/mipmap.cpp
    src/checkerboard.cpp
)
target_link_libraries(hikari-texture PUBLIC hikari-core hikari-spectrum)

add_executable(hikari-texture-test src/test.cpp)
target_compile_features(hikari-texture-test PRIVATE cxx_std_20)
target_compile_options(hikari-texture-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-texture-test   PRIVATE hikari-texture)

set_target_properties(hikari-texture      PROPERTIES FOLDER "src/hikari/texture")
set_target_properties(hikari-texture-test PROPERTIES FOLDER "src/hikari/texture/test")
