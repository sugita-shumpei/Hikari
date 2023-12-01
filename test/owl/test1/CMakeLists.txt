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

add_executable(         hikari-test-owl-test1 hostCode.h deviceCode.h hostCode.cpp)
target_compile_features(hikari-test-owl-test1 PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-test1 PRIVATE hikari-test-owl-testlib hikari-test-owl-test1-ptx owl::owl glfw glm::glm)

set_target_properties(hikari-test-owl-test1         PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-ptx     PROPERTIES FOLDER "test/owl/test1")
set_target_properties(hikari-test-owl-test1-ptx_ptx PROPERTIES FOLDER "test/owl/test1")