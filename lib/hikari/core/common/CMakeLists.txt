find_package(Imath CONFIG REQUIRED)
add_library(hikari-core-common    STATIC)
target_sources(hikari-core-common PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/data_type.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/platform.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/flags.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/spin_lock.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/timer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/common/unique_id.h
${CMAKE_CURRENT_SOURCE_DIR}/src/spin_lock.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/timer.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/unique_id.cpp
)
target_include_directories(hikari-core-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hikari-core-common PUBLIC Imath::Imath Imath::ImathConfig)
target_compile_features(hikari-core-common PRIVATE cxx_std_20)
target_compile_options(hikari-core-common PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::common ALIAS hikari-core-common)
