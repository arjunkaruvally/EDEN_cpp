//
// Created by arjun on 7/18/23.
//
#include <torch/torch.h>
#include <vector>
#include <matplot/matplot.h>
#include <iostream>
#include "linear.h"
#include "progressbar.cpp"


int main() {
    Linear model = Linear();

    torch::Tensor input_signal = 0.1*at::ones({2, 1}, torch::device(torch::kCUDA));
    torch::Tensor interaction_matrix = at::zeros({2, 2}, torch::device(torch::kCUDA));

    interaction_matrix.index_put_({0, 1}, -1);
    interaction_matrix.index_put_({1, 0}, 1);

    model.set_interactions(interaction_matrix);

    std::cout<<"Simulating..."<<std::endl;

    std::vector<float> simulated_x = {};
    std::vector<float> simulated_y = {};

    torch::Tensor cur_state = model.get_state().to(torch::kFloat32);

    progressbar bar(100000);
    for(int i=0; i<=100000; i++){
        bar.update();
        simulated_x.push_back(cur_state[0][0].item<float>());
        simulated_y.push_back(cur_state[1][0].item<float>());

        model.update(input_signal);
        cur_state = model.get_state().to(torch::kFloat32);
        input_signal *= 0;  // no more inputs after the initial state
    }

    matplot::plot(simulated_x, simulated_y, "-o");
    matplot::hold(matplot::on);
    matplot::show();

    std::cout<<"Simulation Complete"<<std::endl;

    return 0;
}
