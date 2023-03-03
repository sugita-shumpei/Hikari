add_executable(Hikari-Test-Test0 Include/Test0.h Src/Test0.cpp)
target_include_directories(Hikari-Test-Test0 PUBLIC Include)
target_link_libraries(Hikari-Test-Test0 Hikari-Base)