find_package(boost REQUIRED)
find_package(nlohmann_json REQUIRED)
add_library(Hikari-Core-Common
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/pch.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKStringUtils.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKBytesUtils.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKArrayUtils.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKInterface.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKMacro.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKUUID.h
	${CMAKE_CURRENT_SOURCE_DIR}/Interface/HKJSON.h
	${CMAKE_CURRENT_SOURCE_DIR}/Src/HKInterface.cpp
)
target_include_directories(Hikari-Core-Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(Hikari-Core-Common PUBLIC boost::boost nlohmann_json::nlohmann_json)