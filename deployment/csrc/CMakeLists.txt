cmake_minimum_required(VERSION 3.10)
project(Deployment VERSION 1.0 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-w)
# find_package(catkin REQUIRED COMPONENTS
#   cv_bridge
#   image_transport
# )
find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)

# set(
#         CUDA_NVCC_FLAGS
#         ${CUDA_NVCC_FLAGS};
#         -O3
#         -gencode arch=compute_75,code=sm_75
# )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(TENSORRT_ROOT /home/sunwenchao/TensorRT-8.4.1.5)
set(CUDA_HOME /usr/local/cuda)
include_directories(${TENSORRT_ROOT}/include ${CUDA_HOME}/include)
link_directories(${TENSORRT_ROOT}/lib ${CUDA_HOME}/lib64)


message("-----------------------------")
message(${TENSORRT_ROOT}/lib)
message(${OpenCV_LIBRARIES})
message(${OpenCV_INCLUDE_DIRS})


add_executable(inference inference.cpp)
target_link_libraries(inference nvinfer nvinfer_plugin cudart )