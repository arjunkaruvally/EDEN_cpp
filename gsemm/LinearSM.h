//
// Created by arjun on 7/18/23.
//

#ifndef GSEMM_CPP_LINEARSM_H
#define GSEMM_CPP_LINEARSM_H

#include <torch/torch.h>
#include "dynSim.h"


struct InteractionMatrices{
    torch::Tensor Xi;
    torch::Tensor Phi;
};

struct States{
    torch::Tensor V_f;
    torch::Tensor V_h;
    torch::Tensor V_d;
};

class LinearSM : public DynamicalSystem{
private:
    States X;
    InteractionMatrices W;
    torch::DeviceType device;

public:
    int N_f;
    int N_h;
    float T_f;
    float T_d;
    float alpha_c;
    float alpha_s;
    double pattern_bias;
    bool PLASTIC;
    float Tl_xi;
    float Tl_phi;
    LinearSM();
    void initialize(torch::DeviceType device=torch::kCUDA, int _N_f=100, int _N_h=4, float _alpha_c=7, float _T_d=20,
                    float _T_f=0.01, float _Tl_xi=20, float _Tl_phi=20, float _alpha_s=1.0);
    void set_interactions(torch::Tensor _Xi, torch::Tensor _Phi);
    void set_sequence_length(int seq_length);
    void update(const torch::Tensor& input_signal);
    void learn(const torch::Tensor& input_signal);
    void reset_states();
    torch::Tensor simulate_system(const torch::Tensor& input_signal, const int simulation_steps = 100000,
                         const int diagnostic_timescale = 50, const int swindow=10, bool VERBOSE=false);

    torch::Tensor f_Vf(const torch::Tensor&);
    torch::Tensor f_Vd(const torch::Tensor&);
    torch::Tensor f_Xi(const torch::Tensor&, const torch::Tensor&);
    torch::Tensor f_Phi(const torch::Tensor&, const torch::Tensor&);
    float get_energy(States& state);
    States get_state();
    void set_V_f(const torch::Tensor signal);
    InteractionMatrices get_interactions();

    int evaluate_capacity(const int max_seq_length, const int diagnostic_timescale, const int swindow, const bool VERBOSE);
    int evaluate_capacity_ST(const int max_seq_length, const int diagnostic_timescale, const int swindow, const int n_samples, const bool VERBOSE);

    std::vector<int> evaluate_learning(const int sequence_length, const int diagnostic_timescale, const int swindow,
                                       const int steps_per_memory, const bool VERBOSE);


    bool evaluate_LT_learning(const int sequence_length, const int diagnostic_timescale, const int swindow,
                                          const int steps_per_memory, const int epochs, const bool VERBOSE,
                                          std::vector<float> &energy_diagnostic,
                                          const int pattern_type=0, const std::string& kDataRoot="data");
};


#endif //GSEMM_CPP_LINEARSM_H
