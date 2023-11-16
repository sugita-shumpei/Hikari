find_package(glfw3 CONFIG REQUIRED)
embed_ptx(
  OUTPUT_TARGET
    hikari-test-owl-test0-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.h
    deviceCode.cu
)

add_executable(         hikari-test-owl-test0 hostCode.h deviceCode.h hostCode.cpp)
target_compile_features(hikari-test-owl-test0 PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-test0 PRIVATE hikari-test-owl-testlib hikari-test-owl-test0-ptx owl::owl glfw)

set_target_properties(hikari-test-owl-test0         PROPERTIES FOLDER "test/owl/test0")
set_target_properties(hikari-test-owl-test0-ptx     PROPERTIES FOLDER "test/owl/test0")
set_target_properties(hikari-test-owl-test0-ptx_ptx PROPERTIES FOLDER "test/owl/test0")