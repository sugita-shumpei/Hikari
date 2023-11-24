add_library(hikari-core SHARED)
target_compile_features(hikari-core PRIVATE cxx_std_17)
target_compile_options(hikari-core PRIVATE "/Zc:__cplusplus")
target_compile_definitions(hikari-core PRIVATE -DHK_MATH_USE_STD_CXX)
target_sources(hikari-core PRIVATE 

interface/hikari/data_type.h
interface/hikari/platform.h
interface/hikari/uuid.h
interface/hikari/object.h 
interface/hikari/object_array.h 
interface/hikari/value_array.h
interface/hikari/ref_ptr.h
interface/hikari/dynamic_loader.h
interface/hikari/plugin.h
interface/hikari/shape.h

include/hikari/ref_cnt_object.h

src/object.cpp 
src/object_array.cpp 
src/value_array.cpp
src/ref_cnt_object.cpp
src/plugin.cpp
src/shape.cpp
)

target_include_directories(hikari-core 
PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/interface 
	${CMAKE_CURRENT_SOURCE_DIR}/../math/interface 
PRIVATE 
	${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_library(hikari::core ALIAS hikari-core)

add_library(hikari-core-headers INTERFACE)
target_include_directories(hikari-core-headers 
INTERFACE 
	${CMAKE_CURRENT_SOURCE_DIR}/interface
	${CMAKE_CURRENT_SOURCE_DIR}/../math/interface 
)
add_library(hikari::core::headers ALIAS hikari-core-headers)

set_target_properties(hikari-core         PROPERTIES FOLDER "src/hikari/core")
set_target_properties(hikari-core-headers PROPERTIES FOLDER "src/hikari/core")

