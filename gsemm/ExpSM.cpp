//
// Created by arjun on 7/18/23.
//

#include "ExpSM.h"
#include "progressbar/progressbar.cpp"

ExpSM::ExpSM(){
    this->PLASTIC = false;
    this->initialize();
}

void ExpSM::initialize(torch::DeviceType device, int _N_f, int _N_h, float _alpha_c, float _T_d, float _T_f,
                       float _Tl_xi, float _Tl_phi, float _alpha_s) {
    this->N_f = _N_f;
    this->N_h = _N_h;
    this->T_f = _T_f;
    this->T_d = _T_d;
    this->Tl_xi = _Tl_xi;
    this->Tl_phi = _Tl_phi;
    this->alpha_c = _alpha_c;
    this->alpha_s = _alpha_s;
    this->device = device;
    this->pattern_bias = 0.0;  // If to draw biased patterns
    this->NO_PHI = false;  // no phi can be used to store sequence patterns with high efficiency

    // Set matrix interactions
    this->set_sequence_length(_N_h);

}

void ExpSM::reset_states() {
    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    this->X.V_f = at::zeros({this->N_f, 1}, opts);
    this->X.V_h = at::zeros({this->N_h, 1}, opts);
    this->X.V_d = at::zeros({this->N_f, 1}, opts);
}

void ExpSM::set_interactions(torch::Tensor _Xi, torch::Tensor _Phi) {
    this->W.Xi = _Xi;
    if(!this->NO_PHI){
        this->W.Phi = _Phi;
    }
}

void ExpSM::set_sequence_length(int seq_length) {
    this->N_h = seq_length;

    if(!this->NO_PHI){
        torch::Tensor phi = at::eye(seq_length).to(torch::kFloat);
        phi = at::roll(phi, 1, 1);
        phi.index_put_({this->N_h-1, 0}, 0);

        torch::Tensor probs = at::full(2, 0.5);
        probs.index_put_({0}, 0.5-this->pattern_bias);
        probs.index_put_({0}, 0.5+this->pattern_bias);

        this->set_interactions((at::multinomial(probs, this->N_f*this->N_h,
                                                true)*2-1).to(torch::kFloat).reshape({this->N_f,
                                                                                      this->N_h}), phi);
    }
    else{
        torch::Tensor phi = at::eye(2).to(torch::kFloat);

        torch::Tensor probs = at::full(2, 0.5);
        probs.index_put_({0}, 0.5-this->pattern_bias);
        probs.index_put_({0}, 0.5+this->pattern_bias);
        this->set_interactions((at::multinomial(probs, this->N_f*this->N_h,
                                                true)*2-1).to(torch::kFloat).reshape({this->N_f,
                                                                                      this->N_h}), phi);
    }

    this->reset_states();
}

States ExpSM::get_state() {
    return this->X;
}

void ExpSM::set_V_f(const torch::Tensor signal){
    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    this->X.V_f = at::zeros({this->N_f, 1}, opts) + signal;
}

InteractionMatrices ExpSM::get_interactions() {
    return this->W;
}

torch::Tensor ExpSM::f_Vf(const torch::Tensor& _x) {
//    std::cout<<"===========================START DEBUG"<<std::endl;
//    std::cout<<torch::transpose(this->W.Xi, 0, 1).device()<<std::endl;
//    std::cout<<_x.device()<<std::endl;
//    std::cout<<"===========================END DEBUG"<<std::endl;

    this->X.V_h = this->alpha_s*torch::matmul(torch::transpose(this->W.Xi, 0, 1), _x);

    using namespace torch::indexing;

    if(this->NO_PHI){
//        std::cout<<this->X.V_h;
//        std::cout<<this->X.V_h.index({Slice(1, None, None)}) +
//                   this->alpha_c * torch::matmul(torch::transpose(this->W.Xi, 0, 1), torch::tanh(this->X.V_d))
//                           .index({Slice(None, -1, None)});
        (this->X.V_h).index_put_({Slice(1, None, None)},
                                 this->X.V_h.index({Slice(1, None, None)}) +
                                 this->alpha_c * torch::matmul(torch::transpose(this->W.Xi, 0, 1),
                                                               torch::tanh(this->X.V_d))
                                 .index({Slice(None, -1, None)}));
//        std::cout<<this->X.V_h;
//        exit(0);
    }
    else{
//        std::cout<<this->X.V_h;
        this->X.V_h += this->alpha_c * torch::matmul(torch::transpose(this->W.Phi, 0, 1),
                                                     torch::matmul(torch::transpose(this->W.Xi, 0, 1),
                                                                   torch::tanh(this->X.V_d)));
//        std::cout<<this->X.V_h;
//        exit(0);
    }
    return (1/this->T_f) * (torch::matmul(this->W.Xi,torch::softmax(this->X.V_h, 0)) - _x);
//    return (1/this->T_f) * _x;
}

torch::Tensor ExpSM::f_Vd(const torch::Tensor& _x) {
    return (1/this->T_d)*(this->X.V_f - _x);
}

torch::Tensor ExpSM::f_Xi(const torch::Tensor& _target_Vf, const torch::Tensor& _target_Vh) {
    auto x = (torch::matmul(_target_Vf,
                            torch::transpose(_target_Vh, 0, 1)));
    x -= (torch::matmul(this->X.V_f,
                        torch::softmax(this->X.V_h, 0).reshape({1, this->N_h})));

    x += sqrt(this->alpha_s) * x;

    x += this->alpha_c * torch::linalg::multi_dot({ this->X.V_d,
                                            torch::transpose(_target_Vh, 0, 1),
                                            torch::transpose(this->W.Phi, 0, 1) });

    x -= this->alpha_c * torch::linalg::multi_dot({ this->X.V_d,
                                            torch::softmax(this->X.V_h, 0).reshape({1, this->N_h}),
                                            torch::transpose(this->W.Phi, 0, 1) });

    return (1/this->Tl_xi)*x;
}

torch::Tensor ExpSM::f_Phi(const torch::Tensor& _target_Vf, const torch::Tensor& _target_Vh) {
    auto x = this->alpha_c * torch::linalg::multi_dot({ torch::transpose(this->W.Xi, 0, 1),
                                                this->X.V_d, torch::transpose(_target_Vh, 0, 1)});
    x -= this->alpha_c * torch::linalg::multi_dot({ torch::transpose(this->W.Xi, 0, 1),
                                            this->X.V_d, torch::softmax(this->X.V_h, 0).reshape({1, this->N_h})});
    return (1/this->Tl_phi)*x;
}

void ExpSM::update(const torch::Tensor& input_signal) {
    this->X.V_f += this->get_step_approximation(this->f_Vf(this->X.V_f) + input_signal,
                                                this->X.V_f);
    this->X.V_d += this->get_step_approximation(this->f_Vd(this->X.V_d), this->X.V_d);
}

void ExpSM::learn(const torch::Tensor& target_signal) {
    auto target_Vf = target_signal.reshape({this->N_f, 1});
    auto target_Vh = sqrt(this->alpha_s)* torch::matmul(torch::transpose(this->W.Xi, 0, 1),
                                                        target_Vf);
    target_Vh = torch::softmax(target_Vh, 0).reshape({this->N_h, 1});

    this->W.Xi += this->get_step_approximation(this->f_Xi(target_Vf, target_Vh),
                                                this->W.Xi);
    this->W.Phi += this->get_step_approximation(this->f_Phi(target_Vf, target_Vh),
                                                this->W.Phi);
}

torch::Tensor ExpSM::simulate_system(const torch::Tensor& input_signal,
                                                       const int simulation_steps,
                                                       const int diagnostic_timescale,
                                                       const int swindow,
                                                       bool VERBOSE){
    const int diagnostic_size = floor(simulation_steps / diagnostic_timescale);

    std::vector<std::vector<float>> mmu(this->N_h,
                                        std::vector<float> (diagnostic_size,
                                                            0));
    progressbar bar(simulation_steps);
    for(int i=0; i<simulation_steps; i++){
        if(VERBOSE){
            bar.update();
        }
        if(i%diagnostic_timescale == 0) {
            auto mmu_computed = torch::matmul(torch::transpose(this->get_interactions().Xi,
                                                               0, 1),
                                              torch::clamp(this->get_state().V_f, -1, 1)) / this->N_h;
            for(int j=0; j<this->N_h; j++){
                mmu[j][floor(i/diagnostic_timescale)] = mmu_computed[j][0].item<float>();
            }
        }
        this->update(input_signal[i]);
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat);
    auto mmu_tensor = torch::zeros({this->N_h, diagnostic_size}, opts);
    for (int i = 0; i < this->N_h; i++)
        mmu_tensor.slice(0, i,i+1) = torch::from_blob(mmu[i].data(), {diagnostic_size}, opts);

    auto output = mmu_tensor.argmax(0);

    if(VERBOSE){
        std::cout<<"output: "<<output.reshape({1, diagnostic_size})<<std::endl;
    }

    auto smoothed_output = torch::zeros(diagnostic_size-swindow,
                                        torch::TensorOptions().dtype(torch::kInt));
    auto sliding_window = torch::zeros(swindow, torch::TensorOptions().dtype(torch::kInt));
    // smooth out the transients
    for(int i=0; i<diagnostic_size-swindow; i++){
        sliding_window = output.index({torch::indexing::Slice(i, i + swindow)});
        smoothed_output.index_put_({i},
                                   sliding_window.index({std::get<2>(at::_unique2(sliding_window,
                                                                                  true, false,
                                                                                  true)).argmax()}));
    }
//    std::cout<<smoothed_output;

    return smoothed_output;
}

bool ExpSM::evaluate_LT_learning(const int sequence_length, const int diagnostic_timescale, const int swindow,
                                 const int steps_per_memory, const int epochs, const bool VERBOSE,
                                 std::vector<float> &energy_diagnostic,
                                 const int pattern_type, const std::string& kDataRoot) {
    int memory_id = 0;
    auto patterns = (at::randint(0, 2, {this->N_f, sequence_length})*2-1).to(torch::kFloat);
//    torch::data::datasets::MapDataset train_dataset;

    if(pattern_type == 1){ // learn MNIST patterns
        assert (this->N_f == 784);
        auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                                                  .map(torch::data::transforms::Stack<>());

        auto batch = train_dataset.get_batch({1, 3, 5, 7, 9, 11, 13, 15, 17, 19});

        patterns = batch.data.reshape({batch.target.sizes()[0], this->N_f}).transpose(0, 1);
        patterns.index_put_({patterns<0.5}, -1);
        patterns.index_put_({patterns>=0.5}, 1);
    }
    auto unique_result = at::_unique2(patterns, true, false, true);
    std::cout<<"Input Statistics=============="<<std::endl;
    std::cout<<"Domain: "<<std::get<0>(unique_result)<<std::endl;
    std::cout<<"Counts: "<<std::get<2>(unique_result)/(sequence_length*this->N_f)<<std::endl;
    std::cout<<"Size of pattern: "<<patterns.sizes()<<std::endl;
    std::cout<<"=============="<<std::endl;

    this->W.Xi = torch::randn({this->N_f, this->N_h});
    this->W.Xi /= this->N_f;
    this->W.Phi = torch::randn({this->N_h, this->N_h});
    this->W.Phi /= this->N_h;

    std::cout<<"Model Statistics=============="<<std::endl;
    std::cout<<"Xi size: "<<this->W.Xi.sizes()<<std::endl;
    std::cout<<"Xi stdev: "<<torch::std(this->W.Xi)<<std::endl;
    std::cout<<"Phi size: "<<this->W.Phi.sizes()<<std::endl;
    std::cout<<"Phi stdev: "<<torch::std(this->W.Phi)<<std::endl;
    std::cout<<"Size of pattern: "<<patterns.sizes()<<std::endl;
    std::cout<<"=============="<<std::endl;

    /*
     Perform learning on the episodic memory task
     1 - episodic memory task (with epochs)
     2 - recall of the list items
    */

    // 1 - episodic memory task
    if(VERBOSE){
        std::cout<<"1 - running the episodic memory task..."<<std::endl;
    }
    progressbar bar(epochs);

    for(int epoch=1; epoch <= epochs; epoch++){
        std::vector<float> energy_epoch;
        if(VERBOSE){
            bar.update();
        }
        this->reset_states();
        for(int i=0; i<steps_per_memory*sequence_length; i++){
            memory_id = floor(i/steps_per_memory);
            auto current_target = patterns.index({torch::indexing::Slice(0, torch::indexing::None),
                                                  torch::indexing::Slice(memory_id, memory_id+1)});
            this->update(current_target);

            //Also learn
            this->learn(current_target);

            if (i%diagnostic_timescale == 0){
                energy_epoch.push_back(this->get_energy(this->X));
            }
        }
        energy_diagnostic.push_back(std::accumulate(energy_epoch.begin(),
                                                    energy_epoch.end(), 0.0) / energy_epoch.size());
    }

    // 2 - recall of the list items
    if(VERBOSE){
        std::cout<<"2 - episodic recall..."<<std::endl;
    }
    // set the V_f to the first pattern
    this->set_V_f(patterns.index({torch::indexing::Slice(0,torch::indexing::None),
                                                     torch::indexing::Slice(0,1)}));
    this->X.V_h = torch::zeros({this->N_h, 1});
    this->X.V_d = torch::zeros({this->N_f, 1});

    int i=0;
    int expected_output = -1;
    int swindow_count = 0;
    int patience = 0;
    auto sliding_window = torch::zeros({swindow});
    auto input_signal = torch::zeros({this->N_f, 1});
    std::vector<int> outputs;
    while(patience < steps_per_memory*1.5){
        if(i%diagnostic_timescale == 0) {
            auto mmu_computed = torch::matmul(torch::transpose(patterns,
                                                               0, 1),
                                              this->get_state().V_f) / this->N_h;
            sliding_window = at::roll(sliding_window, -1);
            sliding_window.index_put_({swindow-1}, torch::argmax(mmu_computed));

            if(i/diagnostic_timescale > swindow){
                int cur_output = sliding_window.index({std::get<2>(at::_unique2(sliding_window,
                                                                                true, false,
                                                                                true)).argmax()}).item<int>();

                if(cur_output != expected_output) {
                    std::cout << "cur: "<<cur_output<<std::endl;
                }

                if(cur_output == expected_output+1){
                    patience = 0;
                    expected_output++;
                    if(expected_output >= sequence_length){
                        return true;
                    }
                }
                else if(cur_output == expected_output){
                    patience++;
                } else {
                    if(VERBOSE){
                        std::cout<<"FAILURE: output is not expected - output: "<<cur_output<<" expected: "<<expected_output<<std::endl;
                    }
                    return false;
                }
            }

        }
        this->update(input_signal);
        i++;
    }
    if(VERBOSE){
        std::cout<<std::endl<<"episodic memory learning evaluation complete"<<std::endl;
    }
    if(expected_output >= sequence_length-1){
        return true;
    } else {
        return false;
    }
}


std::vector<int> ExpSM::evaluate_learning(const int sequence_length, const int diagnostic_timescale, const int swindow,
                              const int steps_per_memory, const bool VERBOSE) {
    int memory_id = 0;

    auto patterns = (at::randint(0, 2, {this->N_f, this->N_h})*2-1).to(torch::kFloat);

    this->W.Xi = torch::randn({this->N_f, this->N_h});
    this->W.Xi /= torch::norm(this->W.Xi, 2);
    this->W.Phi = torch::randn({this->N_h, this->N_h});
    this->W.Phi /= torch::norm(this->W.Phi, 2);

    /*
     Perform learning on the episodic memory task
     1 - episodic memory task
     2 - recall of the list items
    */

    // 1 - episodic memory task
    if(VERBOSE){
        std::cout<<"1 - running the episodic memory task..."<<std::endl;
    }
    progressbar bar(steps_per_memory*sequence_length);
    for(int i=0; i<steps_per_memory*sequence_length; i++){
        if(VERBOSE){
            bar.update();
        }
        memory_id = floor(i/steps_per_memory);
        auto current_target = patterns.index({torch::indexing::Slice(0, torch::indexing::None),
                                                torch::indexing::Slice(memory_id, memory_id+1)});
        this->update(current_target);

        //Also learn
        this->learn(current_target);
    }

    // 2 - recall of the list items
    if(VERBOSE){
        std::cout<<"2 - episodic recall..."<<std::endl;
    }
    // set the V_f to some initial state
    this->set_V_f(torch::randn({this->N_f, 1}));
    this->X.V_h = torch::zeros({this->N_h, 1});
    this->X.V_d = torch::zeros({this->N_f, 1});

    int i=0;
    int expected_output = -1;
    int swindow_count = 0;
    auto sliding_window = torch::zeros({swindow});
    auto input_signal = torch::zeros({this->N_f, 1});
    std::vector<int> outputs;
    progressbar bar1(steps_per_memory*sequence_length*2);
    while(i < steps_per_memory*sequence_length*2){
        if(VERBOSE){
            bar1.update();
        }
        if(i%diagnostic_timescale == 0) {
            auto mmu_computed = torch::matmul(torch::transpose(patterns,
                                                               0, 1),
                                              torch::clamp(this->get_state().V_f, -1, 1)) / this->N_h;

            if(torch::max(mmu_computed).item<float>() > 0.6){
                sliding_window = at::roll(sliding_window, -1);
                sliding_window.index_put_({swindow-1}, torch::argmax(mmu_computed));
                swindow_count++;
            }

            if(swindow_count > swindow){
                int cur_output = sliding_window.index({std::get<2>(at::_unique2(sliding_window,
                                                                                true, false,
                                                                                true)).argmax()}).item<int>();

                if(cur_output != expected_output || expected_output == -1) {
                    if(cur_output < sequence_length){
                        if(outputs.size() == 0 || cur_output != outputs[outputs.size()-1]) {
                            outputs.push_back(cur_output);
                        }
                    }
                    expected_output = cur_output;
                }
            }

        }
        this->update(input_signal);
        i++;
    }
    if(VERBOSE){
        std::cout<<std::endl<<"episodic memory learning evaluation complete"<<std::endl;
    }
    return outputs;
}

int ExpSM::evaluate_capacity(const int max_seq_length, const int diagnostic_timescale, const int swindow,
                             const bool VERBOSE) {
    int lower_npatterns = 2;
    int upper_npatterns = max_seq_length;
    int cur_npatterns;
    bool atleast_one_success = false;
    auto input_signal = torch::zeros({this->N_f, 1});
    auto sliding_window = torch::zeros({swindow});
    while(lower_npatterns < upper_npatterns){
        cur_npatterns = std::floor((upper_npatterns+lower_npatterns)/2);
        std::cout<<"Testing n_patterns: "<<cur_npatterns<<" in range ["<<lower_npatterns<<", "<<upper_npatterns<<"]"<<std::endl;
        this->set_sequence_length(cur_npatterns);

        int patience = 0;
        int i=0;
        int expected_output = 0;
        bool seq_success = false;
        // run simulation here and test directly without saving unecessary run information
        progressbar bar(10000000);

        // Set start state of the system as the first memory
        this->set_V_f(this->get_interactions().Xi.index({torch::indexing::Slice(0,torch::indexing::None),
                                                         torch::indexing::Slice(0,1)}));

        while(patience < 1000){
            if(VERBOSE){
//                bar.update();
            }
//            std::cout<<"i: "<<i<<"patience: "<<patience<<std::endl;

            if(i%diagnostic_timescale == 0) {
                auto mmu_computed = torch::matmul(torch::transpose(this->get_interactions().Xi,
                                                                   0, 1),
                                                  this->get_state().V_f) / this->N_h;
                sliding_window = at::roll(sliding_window, -1);
                sliding_window.index_put_({swindow-1}, torch::argmax(mmu_computed));

                if(i/diagnostic_timescale > swindow){
                    int cur_output = sliding_window.index({std::get<2>(at::_unique2(sliding_window,
                                                                   true, false,
                                                                   true)).argmax()}).item<int>();
//                    std::cout<<std::endl<<"cur_out: "<<cur_output<<std::endl;
//                    std::cout<<sliding_window;

                    if(cur_output != expected_output) {
                        std::cout << "cur: "<<cur_output<<std::endl;
                    }
//                    else {
//                        std::cout << "patience: "<<patience<<" ";
//                    }

                    if(cur_output == expected_output+1){
                        patience = 0;
                        expected_output++;
                    }
                    else if(cur_output == expected_output){
                        patience++;
                    } else {
                        if(VERBOSE){
                            std::cout<<"FAILURE: output is not expected - output: "<<cur_output<<" expected: "<<expected_output<<std::endl;
                        }
                        break;
                    }
                }

            }
            this->update(input_signal);
            i++;
        }
        if(expected_output == cur_npatterns-1){
            seq_success = true;
        }
        // END run simulation

        if(seq_success) {
            lower_npatterns = cur_npatterns+1;
            atleast_one_success = true;
        } else {
            upper_npatterns = cur_npatterns-1;
        }
    }

    if(! atleast_one_success){
        return 0;
    }
    return cur_npatterns;
}

int ExpSM::evaluate_capacity_ST(const int max_seq_length, const int diagnostic_timescale, const int swindow,
                             const int n_samples, const bool VERBOSE) {
    int lower_npatterns = 2;
    int upper_npatterns = max_seq_length;
    int cur_npatterns;
    bool atleast_one_success = false;

    while(lower_npatterns < upper_npatterns){  // binary search for the maximum number of patterns that can be stored
        cur_npatterns = std::floor((upper_npatterns+lower_npatterns)/2);  // Set the current number of patterns
        std::cout<<"Testing n_patterns: "<<cur_npatterns<<" in range ["<<lower_npatterns<<", "<<upper_npatterns<<"]"<<std::endl;
        float p_T = 0;

        for(int j=0; j< n_samples; j++){  // n_samples random draws to calculate the probability of transition
            std::cout << "p_T " << p_T << "/" << j << std::endl;
            this->set_sequence_length(cur_npatterns);
            // Set start state of the system as the first memory
            this->set_V_f(this->get_interactions().Xi.index({torch::indexing::Slice(0,torch::indexing::None),
                                                             torch::indexing::Slice(0,1)}));
            int patience = 0;
            int i=0;
            int expected_output = 0;
            bool seq_success = false;
            auto input_signal = torch::zeros({this->N_f, 1});
            auto sliding_window = torch::zeros({swindow});

            while(patience < 1000){
                if(i%diagnostic_timescale == 0) {
                    auto mmu_computed = torch::matmul(torch::transpose(this->get_interactions().Xi,
                                                                       0, 1),
                                                      this->get_state().V_f) / this->N_h;
                    sliding_window = at::roll(sliding_window, -1);
                    sliding_window.index_put_({swindow-1}, torch::argmax(mmu_computed));

                    if(i/diagnostic_timescale > swindow){
                        int cur_output = sliding_window.index({std::get<2>(at::_unique2(sliding_window,
                                                                                        true, false,
                                                                                        true)).argmax()}).item<int>();

                        if(cur_output != expected_output) {
                            std::cout << "cur: "<<cur_output<<std::endl;
                        }

                        if(cur_output == expected_output+1){
                            patience = 0;
                            if(cur_output >= 2){
                                std::cout << "SUCCESS";
                                seq_success = true;
                                break;
                            }
                            expected_output++;
                        }
                        else if(cur_output == expected_output){
                            patience++;
                        } else {
                            if(VERBOSE){
                                std::cout<<"FAILURE: output is not expected - output: "<<cur_output<<" expected: "<<expected_output<<std::endl;
                            }
                            break;
                        }
                    }

                }
                this->update(input_signal);
                i++;
            }
            if(seq_success) {
                p_T += 1;
            }
        }
        p_T /= n_samples;
        std::cout<<"p_T calculated: "<<p_T<<std::endl;
        if(p_T >= 0.96) {
            lower_npatterns = cur_npatterns+1;
            atleast_one_success = true;
        } else {
            upper_npatterns = cur_npatterns-1;
        }
    }

    if(! atleast_one_success){
        return 0;
    }
    return cur_npatterns;
}

float ExpSM::get_energy(States& state) {
    auto energy = torch::matmul(torch::transpose(state.V_f, 0, 1), state.V_f);
    energy += torch::matmul(torch::transpose(state.V_h, 0, 1), torch::softmax(state.V_h, 0));
    energy -= torch::logsumexp(state.V_h, 0);
    energy -= sqrt(this->alpha_s) * torch::linalg::multi_dot({torch::transpose(state.V_f, 0, 1),
                                                         this->W.Xi,
                                                         torch::softmax(state.V_h, 0)});
    energy -= this->alpha_c * torch::linalg::multi_dot({torch::transpose(state.V_d, 0, 1),
                                                   this->W.Xi,
                                                   torch::transpose(this->W.Phi, 0, 1),
                                                   torch::softmax(state.V_h, 0)});

    return energy.item<float>();
}
