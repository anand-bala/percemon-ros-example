cmake_minimum_required(VERSION 3.5)
project(perception_monitor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(rclcpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(perception_interfaces REQUIRED)
find_package(monitoring_interfaces REQUIRED)
find_package(PerceMon REQUIRED)


set(ament_deps
    # Core dependency
    rclcpp
    # Message deps
    perception_interfaces
    monitoring_interfaces
    std_msgs
)

add_executable(monitor src/monitor.cc)
target_link_libraries(monitor PRIVATE PerceMon::PerceMon)
ament_target_dependencies(monitor PUBLIC ${ament_deps})
target_compile_features(monitor PRIVATE cxx_std_17)

install(TARGETS monitor DESTINATION lib/${PROJECT_NAME})

ament_package()
