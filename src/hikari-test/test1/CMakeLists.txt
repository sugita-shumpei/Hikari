configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test1_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test1_config.h)

add_executable(hikari-test-test1_c11 )
target_compile_features(hikari-test-test1_c11  PUBLIC c_std_11)
target_sources(hikari-test-test1_c11 PUBLIC test1_c11.c ${CMAKE_CURRENT_BINARY_DIR}/test1_config.h)
target_include_directories(hikari-test-test1_c11 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(hikari-test-test1_c11 PRIVATE hikari::headers hikari::graphics::vulkan::headers)

add_executable(hikari-test-test1_cxx11 )
target_compile_features(hikari-test-test1_cxx11  PUBLIC cxx_std_11)
target_compile_options(hikari-test-test1_cxx11 PUBLIC "/Zc:__cplusplus")
target_sources(hikari-test-test1_cxx11 PUBLIC test1_cxx11.cpp ${CMAKE_CURRENT_BINARY_DIR}/test1_config.h)
target_include_directories(hikari-test-test1_cxx11 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(hikari-test-test1_cxx11 PRIVATE hikari::core hikari::shape::headers)
find_package(Vulkan REQUIRED)
add_executable(hikari-test-test1_cxx17 )
target_compile_features(hikari-test-test1_cxx17  PUBLIC cxx_std_17)
target_compile_options(hikari-test-test1_cxx17 PUBLIC "/Zc:__cplusplus")
target_sources(hikari-test-test1_cxx17 PUBLIC test1_cxx17.cpp ${CMAKE_CURRENT_BINARY_DIR}/test1_config.h)
target_include_directories(hikari-test-test1_cxx17 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(hikari-test-test1_cxx17 PRIVATE hikari::headers hikari::graphics::vulkan::headers hikari::graphics::opengl::headers Vulkan::Headers)

add_custom_command(TARGET hikari-test-test1_cxx11 
COMMAND
${CMAKE_COMMAND} -E copy_if_different 
$<TARGET_RUNTIME_DLLS:hikari-test-test1_cxx11> $<TARGET_FILE_DIR:hikari-test-test1_cxx11> 
COMMAND_EXPAND_LISTS)

set_target_properties(hikari-test-test1_c11    PROPERTIES FOLDER "src/hikari-test/test1")
set_target_properties(hikari-test-test1_cxx11  PROPERTIES FOLDER "src/hikari-test/test1")
set_target_properties(hikari-test-test1_cxx17  PROPERTIES FOLDER "src/hikari-test/test1")