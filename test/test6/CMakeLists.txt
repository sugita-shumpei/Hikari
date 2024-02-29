configure_file(test6_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/test6_config.h)
find_package(fmt           CONFIG REQUIRED)
find_package(glm           CONFIG REQUIRED)
find_package(tinyxml2      CONFIG REQUIRED)
find_package(scn           CONFIG REQUIRED)
find_package(ZLIB                 REQUIRED)
find_package(tinyply       CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Imath         CONFIG REQUIRED)
find_package(Matplot++     CONFIG REQUIRED)
find_package(Vulkan               REQUIRED)
find_package(glfw3         CONFIG REQUIRED)
find_package(glad          CONFIG REQUIRED)
find_package(spdlog        CONFIG REQUIRED)
find_package(CUDAToolkit          REQUIRED)
find_package(imgui         CONFIG REQUIRED)
find_package(tsl-robin-map CONFIG REQUIRED)
find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS BS_thread_pool.hpp)
# find_package(Python COMPONENTS Interpreter Development NumPy)
# find_package(pybind11 CONFIG)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS  matplotlibcpp.h )
add_executable(hikari-test-test6
 main.cpp
 hikari/core/application.h
 hikari/core/data_types.h
 hikari/core/flags.h
 hikari/core/platform.h
 hikari/core/singleton.h
 hikari/core/uuid.h
 hikari/console/application.h
 hikari/console/log_system.h
 hikari/console/log_system.cpp
 hikari/devices/application.h
 hikari/thread/spin_lock.h
 hikari/thread/spin_lock.cpp
 hikari/devices/system.h
 hikari/devices/system.cpp
 hikari/devices/window.h
 hikari/devices/window.cpp
 hikari/devices/monitor.h
 hikari/devices/monitor.cpp
 hikari/events/event.h
 hikari/events/event.cpp
 hikari/events/system.h
 hikari/events/system.cpp
 hikari/events/handler.h
 hikari/events/handler.cpp
 hikari/events/target.h
 hikari/events/target.cpp
 hikari/events/queue.h
 hikari/events/queue.cpp
 hikari/events/application_event.h
 hikari/events/application_event.cpp
 hikari/events/window_event.h
 hikari/events/window_event.cpp
)
target_compile_features(hikari-test-test6 PRIVATE cxx_std_20)
target_compile_options(hikari-test-test6 PRIVATE /utf-8 /Zc:__cplusplus)
target_include_directories(hikari-test-test6
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include
  # ${MATPLOTLIB_CPP_INCLUDE_DIRS}
  # ${Python2_INCLUDE_DIRS}
  ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS}
)
target_link_libraries(hikari-test-test6 PRIVATE
  glad::glad
  glm::glm
  tinyxml2::tinyxml2
  scn::scn
  fmt::fmt
  ZLIB::ZLIB
  tinyply
  tinyobjloader::tinyobjloader
  nlohmann_json::nlohmann_json
  Imath::Imath
  Matplot++::cimg
  Matplot++::matplot
  Vulkan::Vulkan
  CUDA::cuda_driver
  spdlog::spdlog
  imgui::imgui
  glfw
  tsl::robin_map
#  Python::Python
#  Python::NumPy
)

add_custom_command(TARGET hikari-test-test6 POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:hikari-test-test6> $<TARGET_FILE_DIR:hikari-test-test6>
COMMAND_EXPAND_LISTS
)
