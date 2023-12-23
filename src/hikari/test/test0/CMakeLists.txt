
add_executable(hikari-test-test0 test0.cpp)
target_compile_features(hikari-test-test0 PRIVATE cxx_std_20)
target_compile_options(hikari-test-test0  PRIVATE /Zc:__cplusplus /utf-8)
target_link_libraries(hikari-test-test0   PRIVATE hikari-core hikari-shape hikari-camera hikari-light hikari-bsdf hikari-film hikari-spectrum hikari-texture)
set_target_properties(hikari-test-test0 PROPERTIES FOLDER "src/hikari/test/test0")