//
// Created by arjun on 7/18/23.
//

#include "linear.h"

Linear::Linear(){
    this->initialize();
}

void Linear::initialize() {
    this->state = at::zeros({2, 1}, torch::device(torch::kCUDA));
    this->set_interactions(at::zeros({2, 2}, torch::device(torch::kCUDA)));
}

torch::Tensor Linear::get_state() {
    return this->state;
}

void Linear::set_interactions(torch::Tensor _interaction) {
    this->A = _interaction;
}

torch::Tensor Linear::f(torch::Tensor _state){
    return torch::matmul(this->A, _state);
}

void Linear::update(torch::Tensor input_signal) {
    this->state += this->get_step_approximation(this->f(this->state), this->state) + input_signal;
}
