cmake_minimum_required(VERSION 3.0)

set(ProjectName "control")
project(${ProjectName})

include(${PROJECT_SOURCE_DIR}/common_helper/cmakes/build_setting.cmake)

# Ros
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

find_package(catkin  REQUIRED COMPONENTS
  roscpp
  usb_cam
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS} )

set(LIBS 
${OpenCV_LIBS} 
${catkin_LIBRARIES}
-lboost_system
)

# 添加源文件
add_executable(${ProjectName} control.cpp)

## CommonHelper
add_subdirectory(${PROJECT_SOURCE_DIR}/common_helper)
target_include_directories(${ProjectName} PUBLIC ${PROJECT_SOURCE_DIR}/common_helper)
target_link_libraries(${ProjectName} CommonHelper)

# 添加 OpenCV 库的头文件路径和库文件路径
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})


# 设置 C++ 编译器标志
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Werror")



# 链接 库
target_link_libraries(${ProjectName}  ${LIBS} )
