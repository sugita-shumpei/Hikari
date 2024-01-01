
find_package(Stb     REQUIRED)
find_package(tinyexr CONFIG REQUIRED)
add_library(hikari-assets-image STATIC)
target_compile_features   (hikari-assets-image PRIVATE cxx_std_20)
target_compile_options    (hikari-assets-image PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-image PUBLIC include PRIVATE ${Stb_INCLUDE_DIR})
target_sources(hikari-assets-image 
PRIVATE 
    include/hikari/assets/image/importer.h
    include/hikari/assets/image/exporter.h
    src/importer.cpp
    src/exporter.cpp
)

target_link_libraries(hikari-assets-image PUBLIC
hikari-core    
unofficial::tinyexr::tinyexr
)

add_executable(hikari-assets-image-test src/test.cpp)
target_compile_features(hikari-assets-image-test PRIVATE cxx_std_20)
target_compile_options(hikari-assets-image-test  PRIVATE /Zc:__cplusplus /utf-8)
target_include_directories(hikari-assets-image-test PRIVATE ${Stb_INCLUDE_DIR} src)
target_link_libraries(hikari-assets-image-test   PRIVATE hikari-assets-image)


set_target_properties(hikari-assets-image       PROPERTIES FOLDER "src/hikari/assets/image")
set_target_properties(hikari-assets-image-test  PROPERTIES FOLDER "src/hikari/assets/image/test")