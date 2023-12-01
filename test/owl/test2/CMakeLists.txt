find_package(glfw3 CONFIG REQUIRED)
find_package(glm   CONFIG REQUIRED)
embed_ptx(
  OUTPUT_TARGET
    hikari-test-owl-test2-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.h
    deviceCode.cu
)

add_executable(         hikari-test-owl-test2 hostCode.h deviceCode.h hostCode.cpp)
target_include_directories(hikari-test-owl-test2 PRIVATE ${Stb_INCLUDE_DIR})
target_compile_features(hikari-test-owl-test2 PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-test2 PRIVATE hikari-test-owl-testlib hikari-test-owl-test2-ptx owl::owl glfw glm::glm)

set_target_properties(hikari-test-owl-test2         PROPERTIES FOLDER "test/owl/test2")
set_target_properties(hikari-test-owl-test2-ptx     PROPERTIES FOLDER "test/owl/test2")
set_target_properties(hikari-test-owl-test2-ptx_ptx PROPERTIES FOLDER "test/owl/test2")