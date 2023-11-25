add_subdirectory(core)
add_subdirectory(math)
add_subdirectory(shape)
add_subdirectory(graphics)

add_library(hikari-headers INTERFACE)
target_include_directories(hikari-headers 
INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/interface
INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/math/interface
INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/shape/interface
INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/graphics/common/interface
)
add_library(hikari::headers ALIAS hikari-headers)