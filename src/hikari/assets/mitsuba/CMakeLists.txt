find_package(ZLIB            REQUIRED)
find_package(Stb             REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
add_library(hikari-assets-mitsuba STATIC)
target_compile_features(hikari-assets-mitsuba PRIVATE cxx_std_20)
target_compile_options (hikari-assets-mitsuba PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-mitsuba PUBLIC include)
target_sources(hikari-assets-mitsuba 
PRIVATE 
    include/hikari/assets/mitsuba/serialized_mesh.h
    include/hikari/assets/mitsuba/serialized_data.h
    include/hikari/assets/mitsuba/spectrum_data.h
    include/hikari/assets/mitsuba/xml_data.h
    src/serialized_mesh.cpp
    src/serialized_data.cpp
    src/spectrum_data.cpp
    src/xml_data.cpp
)
target_link_libraries(hikari-assets-mitsuba PUBLIC
hikari-core
hikari-film
hikari-bsdf
hikari-camera
hikari-light
hikari-spectrum
hikari-texture
hikari-shape
ZLIB::ZLIB
tinyxml2::tinyxml2
)

add_executable(hikari-assets-mitsuba-test src/test.cpp)
target_compile_features(hikari-assets-mitsuba-test PRIVATE cxx_std_20)
target_compile_options(hikari-assets-mitsuba-test  PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-mitsuba-test PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(hikari-assets-mitsuba-test   PRIVATE hikari-assets-mitsuba)

set_target_properties(hikari-assets-mitsuba      PROPERTIES FOLDER "src/hikari/assets/mitsuba")
set_target_properties(hikari-assets-mitsuba-test PROPERTIES FOLDER "src/hikari/assets/mitsuba/test")
