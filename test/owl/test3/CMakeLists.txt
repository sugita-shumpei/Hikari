find_package(glfw3 CONFIG REQUIRED)
find_package(glm   CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
embed_ptx(
  OUTPUT_TARGET
    hikari-test-owl-test3-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.h
    deviceCode.cu
)

add_executable(         hikari-test-owl-test3 hostCode.h deviceCode.h hostCode.cpp)
target_compile_features(hikari-test-owl-test3 PUBLIC  cxx_std_20)
target_link_libraries(  hikari-test-owl-test3 PRIVATE 
    hikari-test-owl-testlib 
    hikari-test-owl-test3-ptx 
    hikari-assets-mitsuba
    owl::owl 
    glm::glm
    imgui::imgui
)

set_target_properties(hikari-test-owl-test3         PROPERTIES FOLDER "test/owl/test3")
set_target_properties(hikari-test-owl-test3-ptx     PROPERTIES FOLDER "test/owl/test3")
set_target_properties(hikari-test-owl-test3-ptx_ptx PROPERTIES FOLDER "test/owl/test3")
