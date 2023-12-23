//
// Created by arjun on 7/18/23.
//

#ifndef GSEMM_CPP_LINEAR_H
#define GSEMM_CPP_LINEAR_H

#include <torch/torch.h>
#include "dynSim.h"

class Linear: public DynamicalSystem {
private:
    torch::Tensor state;
    torch::Tensor A;
public:
    Linear();

    void initialize();
    void set_interactions(torch::Tensor);
    void update(torch::Tensor input_signal);

    torch::Tensor f(torch::Tensor);
    torch::Tensor get_state();
};


#endif //GSEMM_CPP_LINEAR_H
