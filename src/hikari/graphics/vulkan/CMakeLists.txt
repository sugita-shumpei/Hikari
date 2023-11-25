find_package(Vulkan REQUIRED)

add_library(hikari-graphics-vulkan SHARED)
target_compile_features(hikari-graphics-vulkan PRIVATE cxx_std_17)
target_compile_options(hikari-graphics-vulkan PRIVATE "/Zc:__cplusplus")
target_sources(hikari-graphics-vulkan PRIVATE 
interface/hikari/graphics/vulkan/common.h
interface/hikari/graphics/vulkan/entry.h
interface/hikari/graphics/vulkan/instance.h
interface/hikari/graphics/vulkan/plugin.h
src/entry.cpp
src/instance.cpp
src/plugin.cpp
)

target_include_directories(hikari-graphics-vulkan 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../common/interface
${CMAKE_CURRENT_SOURCE_DIR}/../../core/interface
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
)

target_link_libraries(hikari-graphics-vulkan PRIVATE hikari::graphics::common hikari::core Vulkan::Headers)
add_library(hikari::graphics::vulkan ALIAS hikari-graphics-vulkan)

add_library(hikari-graphics-vulkan-headers INTERFACE)
target_include_directories(hikari-graphics-vulkan-headers 
INTERFACE 
${CMAKE_CURRENT_SOURCE_DIR}/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../common/include
${CMAKE_CURRENT_SOURCE_DIR}/../../core/interface 
${CMAKE_CURRENT_SOURCE_DIR}/../../math/interface 
)
add_library(hikari::graphics::vulkan::headers ALIAS hikari-graphics-vulkan-headers)

set_target_properties(hikari-graphics-vulkan         PROPERTIES FOLDER "src/hikari/graphics/vulkan")
set_target_properties(hikari-graphics-vulkan-headers PROPERTIES FOLDER "src/hikari/graphics/vulkan")

