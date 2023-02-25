add_executable(
Hikari-Test-Test2	
${CMAKE_CURRENT_SOURCE_DIR}/Include/Test2.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/Test2.cpp

)
target_include_directories(Hikari-Test-Test2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(Hikari-Test-Test2 PUBLIC Hikari-Base)