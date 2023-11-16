add_library(hikari-test-test0 SHARED)
target_compile_features(hikari-test-test0 PRIVATE cxx_std_17)
target_compile_options(hikari-test-test0 PRIVATE "/Zc:__cplusplus")
target_sources(hikari-test-test0 PRIVATE test0.h test0.cpp)
target_include_directories(hikari-test-test0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hikari-test-test0 PUBLIC hikari::hikari)

add_executable(hikari-test-test0_c11 )
target_compile_features(hikari-test-test0_c11  PUBLIC c_std_11)
target_sources(hikari-test-test0_c11 PUBLIC test0_c11.c)
target_link_libraries(hikari-test-test0_c11 PRIVATE hikari-test-test0)

add_executable(hikari-test-test0_cxx11 )
target_compile_features(hikari-test-test0_cxx11  PUBLIC cxx_std_11)
target_compile_options(hikari-test-test0_cxx11 PUBLIC "/Zc:__cplusplus")
target_sources(hikari-test-test0_cxx11 PUBLIC test0_cxx11.cpp)
target_link_libraries(hikari-test-test0_cxx11 PRIVATE hikari-test-test0)

add_executable(hikari-test-test0_cxx17 )
target_compile_features(hikari-test-test0_cxx17  PUBLIC cxx_std_17)
target_compile_options(hikari-test-test0_cxx17 PUBLIC "/Zc:__cplusplus")
target_sources(hikari-test-test0_cxx17 PUBLIC test0_cxx17.cpp)
target_link_libraries(hikari-test-test0_cxx17 PRIVATE hikari-test-test0)

add_custom_command(TARGET hikari-test-test0_c11 
COMMAND
${CMAKE_COMMAND} -E copy_if_different 
$<TARGET_RUNTIME_DLLS:hikari-test-test0_c11> $<TARGET_FILE_DIR:hikari-test-test0_c11> 
COMMAND_EXPAND_LISTS)

add_custom_command(TARGET hikari-test-test0_cxx11 
COMMAND
${CMAKE_COMMAND} -E copy_if_different 
$<TARGET_RUNTIME_DLLS:hikari-test-test0_cxx11> $<TARGET_FILE_DIR:hikari-test-test0_cxx11> 
COMMAND_EXPAND_LISTS)

add_custom_command(TARGET hikari-test-test0_cxx17 
COMMAND
${CMAKE_COMMAND} -E copy_if_different 
$<TARGET_RUNTIME_DLLS:hikari-test-test0_cxx17> $<TARGET_FILE_DIR:hikari-test-test0_cxx17> 
COMMAND_EXPAND_LISTS)


set_target_properties(hikari-test-test0        PROPERTIES FOLDER "src/hikari-test/test0")
set_target_properties(hikari-test-test0_c11    PROPERTIES FOLDER "src/hikari-test/test0")
set_target_properties(hikari-test-test0_cxx11  PROPERTIES FOLDER "src/hikari-test/test0")
set_target_properties(hikari-test-test0_cxx17  PROPERTIES FOLDER "src/hikari-test/test0")