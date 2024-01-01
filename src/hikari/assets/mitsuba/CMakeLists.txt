find_package(ZLIB                 REQUIRED)
find_package(Stb                  REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(glad          CONFIG REQUIRED)
find_package(glfw3         CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
add_library(hikari-assets-mitsuba STATIC)
target_compile_features(hikari-assets-mitsuba PRIVATE cxx_std_20)
target_compile_options (hikari-assets-mitsuba PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-mitsuba PUBLIC include)
target_sources(hikari-assets-mitsuba 
PRIVATE 
    include/hikari/assets/mitsuba/scene_importer.h
    src/scene_importer.cpp

    src/serialized_data.h
    src/serialized_data.cpp
    src/spectrum_data.h
    src/spectrum_data.cpp
    src/xml_data.h
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
hikari-assets-image
ZLIB::ZLIB
tinyxml2::tinyxml2
tinyobjloader::tinyobjloader
tinyply
)

add_executable(hikari-assets-mitsuba-test src/test.cpp)
target_compile_features(hikari-assets-mitsuba-test PRIVATE cxx_std_20)
target_compile_options(hikari-assets-mitsuba-test  PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-mitsuba-test PRIVATE ${Stb_INCLUDE_DIR} src)
target_link_libraries(hikari-assets-mitsuba-test   PRIVATE hikari-assets-mitsuba)


add_executable(hikari-assets-mitsuba-test2 src/test2.cpp)
target_compile_features(hikari-assets-mitsuba-test2 PRIVATE cxx_std_20)
target_compile_options(hikari-assets-mitsuba-test2  PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-mitsuba-test2 PRIVATE ${Stb_INCLUDE_DIR} src)
target_link_libraries(hikari-assets-mitsuba-test2   PRIVATE hikari-assets-mitsuba glad::glad glfw)


set_target_properties(hikari-assets-mitsuba       PROPERTIES FOLDER "src/hikari/assets/mitsuba")
set_target_properties(hikari-assets-mitsuba-test  PROPERTIES FOLDER "src/hikari/assets/mitsuba/test")
set_target_properties(hikari-assets-mitsuba-test2 PROPERTIES FOLDER "src/hikari/assets/mitsuba/test")
