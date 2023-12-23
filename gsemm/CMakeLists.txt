find_package(Torch REQUIRED)
add_subdirectory(progressbar)

add_library(gsemm dynSim.cpp linear.cpp ExpSM.cpp LinearSM.cpp progressbar)
target_link_libraries(gsemm "${TORCH_LIBRARIES}")
