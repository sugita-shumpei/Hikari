find_package(CUDAToolkit REQUIRED)
find_package(glfw3       CONFIG REQUIRED)
find_package(glad        CONFIG REQUIRED)
find_package(glm         CONFIG REQUIRED)
find_package(mikktspace  CONFIG REQUIRED)
find_package(tinyobjloader  CONFIG REQUIRED)
find_package(imgui          CONFIG REQUIRED)

set(HK_TESTLIB_ASSETS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../data)
configure_file(testlib_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/testlib_config.h)

add_library(hikari-test-owl-testlib-cuda STATIC)
target_sources(hikari-test-owl-testlib-cuda  PRIVATE 
cuda/tonemap_impl.h
cuda/tonemap_impl.cu
)
target_compile_features(hikari-test-owl-testlib-cuda PRIVATE cuda_std_17)
target_link_directories(hikari-test-owl-testlib-cuda PUBLIC CUDA::cudart_static)

add_library(hikari-test-owl-testlib STATIC)
target_sources(hikari-test-owl-testlib  PRIVATE 
gl_viewer.h 
gl_viewer.cpp 
obj_model.h 
obj_model.cpp 
pinhole_camera.h 
pinhole_camera.cpp
tonemap.h
tonemap.cpp
)

target_include_directories(hikari-test-owl-testlib  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(hikari-test-owl-testlib PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-testlib 
PUBLIC 
CUDA::cudart_static 
glm::glm 
PRIVATE 
glfw
glad::glad 
imgui::imgui 
tinyobjloader::tinyobjloader
mikktspace::mikktspace
hikari-test-owl-testlib-cuda
)

set_target_properties(hikari-test-owl-testlib      PROPERTIES FOLDER "test/owl/testlib")
set_target_properties(hikari-test-owl-testlib-cuda PROPERTIES FOLDER "test/owl/testlib")
