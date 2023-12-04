find_package(glfw3 CONFIG REQUIRED)
find_package(glm   CONFIG REQUIRED)
embed_ptx(
  OUTPUT_TARGET
    hikari-test-owl-test1-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.h
    deviceCode.cu
)

embed_ptx(
  OUTPUT_TARGET
    hikari-test-owl-test1-kernel-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    kernelCode.h
    kernelCode.cu
)

find_package(CUDAToolkit REQUIRED)
add_library(hikari-test-owl-test1-thrust STATIC)
target_sources(hikari-test-owl-test1-thrust PRIVATE thrustCode.h thrustCode.cu)
target_compile_features(hikari-test-owl-test1-thrust PUBLIC  cuda_std_17)
target_link_libraries(hikari-test-owl-test1-thrust   PRIVATE CUDA::cudart_static)

add_executable(         hikari-test-owl-test1 hostCode.h deviceCode.h kernelCode.h hostCode.cpp)
target_include_directories(hikari-test-owl-test1 PRIVATE ${Stb_INCLUDE_DIR})
target_compile_features(hikari-test-owl-test1 PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-test1 PRIVATE 
hikari-test-owl-testlib 
hikari-test-owl-test1-thrust 
hikari-test-owl-test1-ptx 
hikari-test-owl-test1-kernel-ptx 
owl::owl 
glfw 
glm::glm)

set_target_properties(hikari-test-owl-test1         PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-thrust  PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-ptx     PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-ptx_ptx PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-kernel-ptx     PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-kernel-ptx_ptx PROPERTIES FOLDER "test/owl/test1")