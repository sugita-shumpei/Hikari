find_package(glm   CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
add_library(hikari-film STATIC)
target_compile_features(hikari-film PRIVATE cxx_std_20)
target_compile_options(hikari-film PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-film PUBLIC include)
target_sources(hikari-film PRIVATE
    include/hikari/film/hdr.h
    include/hikari/film/spec.h
    src/hdr.cpp
    src/spec.cpp
)
target_link_libraries(hikari-film PUBLIC hikari-core)

add_executable(hikari-film-test src/test.cpp)
target_compile_features(hikari-film-test PRIVATE cxx_std_20)
target_compile_options(hikari-film-test  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-film-test   PRIVATE hikari-film)

set_target_properties(hikari-film      PROPERTIES FOLDER "src/hikari/film")
set_target_properties(hikari-film-test PROPERTIES FOLDER "src/hikari/film/test")
