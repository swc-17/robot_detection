cmake_minimum_required(VERSION 3.10)
project(plugin LANGUAGES CXX C)

set(PLUGIN_ROOT ${PROJECT_SOURCE_DIR})
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    #set(CMAKE_CXX_COMPILER g++) #there exist a conflict between libtorch and gtest if we use cxx
    #set(CMAKE_C_COMPILER gcc)
    message("CMake is set to Debug mode!")
    set(BUILD_GTEST ON)
    set(USE_LIBTORCH OFF)
else(CMAKE_BUILD_TYPE MATCHES "Release")
    set(BUILD_GTEST OFF)
    set(USE_LIBTORCH OFF)
endif()



# plugins
set(PLUGIN_LISTS
    #batchedNMSPlugin
    efficientNMSPlugin
)

# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -ggdb -g -O0 -Wno-deprecated-declarations")
    else(CMAKE_BUILD_TYPE MATCHES "Release")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O2 -Wno-deprecated-declarations -Wno-unused-function")
    endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#--------------------------------------------------
# Version information
#--------------------------------------------------

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------
# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_MYELIN myelin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
#find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
#if(NOT TENSORRT_FOUND)
    #  message(ERROR "Cannot find TensorRT library.")
    #endif()

# --------------------------------
# Plugin library
# --------------------------------
find_package(CUDA 10.2 REQUIRED)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11 -g -G")
else(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11 -O2")
endif()

CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})

if(BUILD_GTEST)
    find_package(GTest REQUIRED)
    if(GTest_FOUND)
        message("Found GTest!")
    endif()

    find_package(Torch REQUIRED)
    if(Torch_FOUND AND USE_LIBTORCH)
        #set(CMAKE_CXX_STANDARD 14)
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        message("Torch libs:${TORCH_LIBRARIES}")
        message("CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")
    endif()
endif()

include_directories(".")

foreach(PLUGIN_NAME ${PLUGIN_LISTS})
    add_subdirectory(${PLUGIN_NAME})
    #list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}") 
    message(STATUS "************${PLUGIN_SOURCES}")
    message(STATUS "************${PLUGIN_CU_SOURCES}")
    
    CUDA_ADD_LIBRARY(${PLUGIN_NAME}_cu SHARED ${PLUGIN_CU_SOURCES})
    target_include_directories(${PLUGIN_NAME}_cu PUBLIC ${INCLUDE_DIR} ${CUDA_INCLUDE_DIRS}  ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} "/usr/local/cuda/include/thrust/system/cuda/detail")
    
    add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SOURCES})
    target_include_directories(${PLUGIN_NAME} PUBLIC ${INCLUDE_DIR} ${CUDA_INCLUDE_DIRS}  ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
    
    target_link_libraries(${PLUGIN_NAME} ${TENSORRT_LIBRARY} )
#${PLUGIN_NAME}_cu) 
endforeach(PLUGIN_NAME)


#foreach(PLUGIN_NAME ${PLUGIN_LISTS})
#   # add_subdirectory(${PLUGIN_NAME})
#    foreach(CU_LISTS ${CU_SOURCES})
#        string(FIND ${CU_LISTS} ${PLUGIN_NAME} pos)
#        if(NOT pos EQUAL -1)
#            message("Found CU_SRCS:${CU_LISTS} of ${PLUGIN_NAME}")
#            # dynamic lib
#            CUDA_ADD_LIBRARY(${PLUGIN_NAME} SHARED ${CU_LISTS})
#            if(NOT ${CU_EXTRA} EQUAL -1)
#                message("Found CU_EXTRA:${CU_EXTRA}")
#               CUDA_ADD_LIBRARY(${PLUGIN_NAME}_extra SHARED ${CU_EXTRA})
#                target_include_directories(${PLUGIN_NAME} PUBLIC ${INCLUDE_DIR})
#               target_link_libraries(${PLUGIN_NAME} ${PLUGIN_NAME}_extra)
#            endif()
#            target_include_directories(${PLUGIN_NAME} PUBLIC ${CUDA_INCLUDE_DIRS}  ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
#            target_link_libraries(${PLUGIN_NAME} ${TENSORRT_LIBRARY})
#        endif()
#    endforeach(CU_LISTS)
#endforeach(PLUGIN_NAME)
