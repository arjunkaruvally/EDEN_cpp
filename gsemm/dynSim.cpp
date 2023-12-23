//
// Created by arjun on 7/18/23.
//
#include <stdexcept>
#include "dynSim.h"

// Constructor
DynamicalSystem::DynamicalSystem(float step_size, ApproximationMethod approximation) {
    this->step_size = step_size;
    this->approximation = approximation;
}

torch::Tensor DynamicalSystem::get_step_approximation(torch::Tensor flow, torch::Tensor param) {
    if(this->approximation == ApproximationMethod::euler) {
        return this->step_size * flow;
    }
    else {
        throw std::invalid_argument("approximation method not implemented");
    }
}

void update(const torch::Tensor& input_signal){
    throw std::invalid_argument("update method not implemented");
}
