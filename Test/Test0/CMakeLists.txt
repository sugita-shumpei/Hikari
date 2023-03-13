add_executable(Hikari-Test-Test0)
target_sources(Hikari-Test-Test0 PUBLIC 
"${CMAKE_CURRENT_SOURCE_DIR}/Include/Test0.h" 
"${CMAKE_CURRENT_SOURCE_DIR}/Src/Test0.cpp" 
)
target_include_directories(Hikari-Test-Test0 PUBLIC
"${CMAKE_CURRENT_SOURCE_DIR}/Include"
)