find_package(function2 CONFIG REQUIRED)
add_library(hikari-core-event    STATIC)
target_sources(hikari-core-event PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/event/handler.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/event/event.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/event/manager.h
${CMAKE_CURRENT_SOURCE_DIR}/include/hikari/core/event/target.h
${CMAKE_CURRENT_SOURCE_DIR}/src/handler.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/event.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/manager.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/target.cpp
)
target_include_directories(hikari-core-event PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hikari-core-event PUBLIC hikari-core-common function2::function2)
target_compile_features(hikari-core-event PRIVATE cxx_std_20)
target_compile_options(hikari-core-event PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus> $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
add_library(hikari::core::event ALIAS hikari-core-event)
