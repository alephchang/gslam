cmake_minimum_required( VERSION 2.8 )
project ( gslam )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O0" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

#message("cmake_module_path: " ${CMAKE_MODULE_PATH})
#message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)

# OpenCV
find_package( OpenCV 3.1 REQUIRED )
find_package(G2O REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Sophus 
find_package(Sophus REQUIRED)

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    g2o_core g2o_stuff g2o_types_sba
    ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
)
############### dependencies ######################
include_directories( ${PROJECT_SOURCE_DIR}/include 
                        ${PROJECT_SOURCE_DIR})
add_subdirectory( src )
add_subdirectory( test )
