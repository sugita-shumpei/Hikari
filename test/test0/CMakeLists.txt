configure_file(test0_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test0_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
add_executable(hikari-test-test0
main.cpp
hikari/core/data_type.h
hikari/core/json.h
hikari/core/json.cpp

hikari/core/json.h
hikari/core/json.cpp
hikari/core/object.h
hikari/core/object.cpp

hikari/core/scene.h
hikari/core/scene.cpp
hikari/core/transform.h
hikari/core/transform.cpp

hikari/assets/mitsuba/scene_importer.h
hikari/assets/mitsuba/scene_importer.cpp
hikari/assets/mitsuba/xml/object.h
hikari/assets/mitsuba/xml/object.cpp
hikari/assets/mitsuba/xml/transform.h
hikari/assets/mitsuba/xml/transform.cpp
hikari/assets/mitsuba/xml/spectrum.h
hikari/assets/mitsuba/xml/spectrum.cpp
hikari/assets/mitsuba/xml/texture.h
hikari/assets/mitsuba/xml/texture.cpp
hikari/assets/mitsuba/xml/bsdf.h
hikari/assets/mitsuba/xml/bsdf.cpp
hikari/assets/mitsuba/xml/shape.h
hikari/assets/mitsuba/xml/shape.cpp
hikari/assets/mitsuba/xml/medium.h
hikari/assets/mitsuba/xml/medium.cpp
hikari/assets/mitsuba/xml/context.h
hikari/assets/mitsuba/xml/context.cpp
hikari/assets/mitsuba/xml/phase.h
hikari/assets/mitsuba/xml/phase.cpp
hikari/assets/mitsuba/xml/volume.h
hikari/assets/mitsuba/xml/volume.cpp
hikari/assets/mitsuba/xml/emitter.h
hikari/assets/mitsuba/xml/emitter.cpp
hikari/assets/mitsuba/xml/film.h
hikari/assets/mitsuba/xml/film.cpp
hikari/assets/mitsuba/xml/rfilter.h
hikari/assets/mitsuba/xml/rfilter.cpp
hikari/assets/mitsuba/xml/sampler.h
hikari/assets/mitsuba/xml/sampler.cpp
hikari/assets/mitsuba/xml/sensor.cpp
hikari/assets/mitsuba/xml/integrator.h
hikari/assets/mitsuba/xml/integrator.cpp
hikari/assets/mitsuba/xml/scene.h
hikari/assets/mitsuba/xml/scene.cpp
hikari/assets/mitsuba/spectrum/serialized_data.h
hikari/assets/mitsuba/spectrum/serialized_data.cpp
hikari/assets/mitsuba/shape/mesh_data.h
hikari/assets/mitsuba/shape/mesh_data.cpp
hikari/assets/mitsuba/shape/obj/mesh_importer.h
hikari/assets/mitsuba/shape/obj/mesh_importer.cpp
hikari/assets/mitsuba/shape/ply/mesh_importer.h
hikari/assets/mitsuba/shape/ply/mesh_importer.cpp
hikari/assets/mitsuba/shape/serialized/mesh_importer.h
hikari/assets/mitsuba/shape/serialized/mesh_importer.cpp
)
target_compile_features(hikari-test-test0 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test0 PRIVATE /utf-8)
target_include_directories(hikari-test-test0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(hikari-test-test0 PRIVATE glm::glm tinyxml2::tinyxml2 scn::scn fmt::fmt ZLIB::ZLIB tinyply tinyobjloader::tinyobjloader nlohmann_json::nlohmann_json)
