add_executable(
Hikari-Test-Test1	
${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1.cpp

)
target_include_directories(Hikari-Test-Test1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(Hikari-Test-Test1 PUBLIC Hikari-Base)