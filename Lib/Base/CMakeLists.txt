find_package(boost REQUIRED)
add_library(Hikari-Base STATIC 
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/Mat.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/Qua.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/Vec.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/Random.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/Constants.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math/HashUtils.h

${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Preprocessors.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Math.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/SpinLock.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/ThreadPool.h
${CMAKE_CURRENT_SOURCE_DIR}/Include/Hikari/Utility.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/SpinLock.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Src/ThreadPool.cpp)
target_include_directories(Hikari-Base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(Hikari-Base PUBLIC boost::boost)