//
// Created by arjun on 7/18/23.
//
#include <bits/stdc++.h>
#include <torch/torch.h>

#ifndef GSEMM_CPP_DYNSIM_H
#define GSEMM_CPP_DYNSIM_H

enum ApproximationMethod {
    range_kutta,
    euler,
    Default = euler
};

class DynamicalSystem{
public:
    float step_size;
    ApproximationMethod approximation;
    DynamicalSystem(float step_size=0.001, ApproximationMethod approximation = Default);

    torch::Tensor get_step_approximation(torch::Tensor flow, torch::Tensor param);
    void update(const torch::Tensor& input_signal);
};


#endif //GSEMM_CPP_DYNSIM_H
