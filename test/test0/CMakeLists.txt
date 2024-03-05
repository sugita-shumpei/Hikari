add_executable(hikari-test-test0 main.cpp)
target_compile_features(hikari-test-test0 PRIVATE cxx_std_20)
target_compile_options(hikari-test-test0 PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(hikari-test-test0 PUBLIC hikari::core::app  hikari::core::ui hikari::core::graphics::vulkan hikari::core::graphics::opengl)
