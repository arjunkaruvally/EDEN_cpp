cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GSEMM)

set(CMAKE_CUDA_ARCHITECTURES 0)

find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CUDA_ARCHITECTURES "Ampere")

# add gsemm subdirectory to the project
add_subdirectory(gsemm)
add_subdirectory(matplotplusplus)
add_subdirectory(argparse)
#add_subdirectory(progressbar)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

## Test linear dynamical system - Experiemnt 0_1
add_executable(0_1_linear experiments/0_linear/1_linear.cpp)
target_link_libraries(0_1_linear "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 0_1_linear PROPERTY CXX_STANDARD 17)

target_include_directories(0_1_linear PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/progressbar"
        "${PROJECT_SOURCE_DIR}/argparse"
        )

## Test Exp-SM model - Experiment 1_0
add_executable(1_0_expSM_testRun experiments/1_expSM/1_expSM_test_run.cpp)
target_link_libraries(1_0_expSM_testRun "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 1_0_expSM_testRun PROPERTY CXX_STANDARD 17)

target_include_directories(1_0_expSM_testRun PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
        )

## Test Linear-SM model - Experiment 1_0
add_executable(1_0_linearSM_testRun experiments/1_expSM/1_linearSM_test_run.cpp)
target_link_libraries(1_0_linearSM_testRun "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 1_0_linearSM_testRun PROPERTY CXX_STANDARD 17)

target_include_directories(1_0_linearSM_testRun PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
)

## Exp-SM LEARN model - Experiment 2_0
#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/2_expSM_learning")
add_executable(2_0_expSM_learning experiments/1_expSM/2_expSM_learning.cpp)
target_link_libraries(2_0_expSM_learning "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 2_0_expSM_learning PROPERTY CXX_STANDARD 17)

target_include_directories(2_0_expSM_learning PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
        )

## Exp-SM LEARN model - Experiment 3_0
#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/2_expSM_learning")
add_executable(3_0_expSM_LT_learning experiments/1_expSM/3_0_expSM_LT_learning.cpp)
target_link_libraries(3_0_expSM_LT_learning "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 3_0_expSM_LT_learning PROPERTY CXX_STANDARD 17)

target_include_directories(3_0_expSM_LT_learning PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
        )

## Exp-SM LEARN model - Experiment 3_0
#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/2_expSM_learning")
add_executable(3_0_1_expSM_LT_learning experiments/1_expSM/3_0_1_expSM_LT_multiple_seeds.cpp)
target_link_libraries(3_0_1_expSM_LT_learning "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 3_0_1_expSM_LT_learning PROPERTY CXX_STANDARD 17)

target_include_directories(3_0_1_expSM_LT_learning PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
)

## Exp-SM LEARN model - Experiment 3_1
#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/2_expSM_learning")
add_executable(3_1_expSM_LT_MNIST experiments/1_expSM/3_1_expSM_LT_MNIST.cpp)
target_link_libraries(3_1_expSM_LT_MNIST "${TORCH_LIBRARIES}" gsemm matplot argparse)

set_property(TARGET 3_1_expSM_LT_MNIST PROPERTY CXX_STANDARD 17)

target_include_directories(3_1_expSM_LT_MNIST PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gsemm"
        "${PROJECT_SOURCE_DIR}/argparse"
        )
