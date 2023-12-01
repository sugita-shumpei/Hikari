find_package(glfw3 CONFIG REQUIRED)
find_package(glad  CONFIG REQUIRED)
find_package(tinyobjloader  CONFIG REQUIRED)
add_library(hikari-test-owl-testlib STATIC)
target_sources(hikari-test-owl-testlib  PRIVATE gl_viewer.h gl_viewer.cpp obj_model.h obj_model.cpp pinhole_camera.h pinhole_camera.cpp)
target_include_directories(hikari-test-owl-testlib  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(hikari-test-owl-testlib PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-testlib PUBLIC owl::owl PRIVATE glad::glad tinyobjloader::tinyobjloader)


set_target_properties(hikari-test-owl-testlib         PROPERTIES FOLDER "test/owl/testlib")