cmake_minimum_required(VERSION 3.0.2)
project(serial_comm)
add_definitions(-std=c++11)
add_definitions(-w)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  serial
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(serialComm serialport.cpp)
add_executable(serialCommWithoutVelocity serialport_without_velocity.cpp)
target_link_libraries(serialComm ${catkin_LIBRARIES})
target_link_libraries(serialCommWithoutVelocity ${catkin_LIBRARIES})

