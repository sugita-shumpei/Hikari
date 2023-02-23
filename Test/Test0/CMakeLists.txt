find_package(glm REQUIRED)
find_package(boost REQUIRED)
add_executable(Hikari-Test-Test0
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test0.h
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test0.cpp
)
target_include_directories(Hikari-Test-Test0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(Hikari-Test-Test0 PUBLIC glm::glm Hikari-Core-Common)