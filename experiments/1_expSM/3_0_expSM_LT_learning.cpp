//
// Created by arjun on 7/19/23.
//
#include <fstream>
#include <torch/torch.h>
#include <vector>
#include <matplot/matplot.h>
#include <argparse/argparse.hpp>
#include <iostream>
#include "ExpSM.h"
#include "kmp_search.h"
#include <filesystem>
namespace fs = std::filesystem;


int main(int argc, char *argv[]) {
    // argparser section
    argparse::ArgumentParser program("Exp SM learning run");

    program.add_argument("--seed")
            .help("Experiment seed.")
            .default_value(0).scan<'i', int>();

    program.add_argument("--experiment_id")
            .default_value(std::string{"0"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
            .help("Id string for the experiment");

    program.add_argument("--n_neurons")
            .help("number of neurons in the network.")
            .default_value(100).scan<'i', int>();

    program.add_argument("--epochs")
            .help("number of epochs to run the sequence.")
            .default_value(10).scan<'i', int>();

    program.add_argument("--diagnostic_timescale")
            .help("The timescale between two diagnostic routines.")
            .default_value(50).scan<'i', int>();

    program.add_argument("--max_seq_length")
            .help("Maximum memory sequence length to test.")
            .default_value(50).scan<'i', int>();

    program.add_argument("--seq_length")
            .help("sequence length to learn.")
            .default_value(10).scan<'i', int>();

    program.add_argument("--steps_per_memory")
            .help("how many steps to show a single memory for.")
            .default_value(1200).scan<'i', int>();

    program.add_argument("--alpha_c")
            .help("alpha_c parameter of the model.")
            .default_value(3.0).scan<'g', double>();

    program.add_argument("--T_d")
            .help("timescale of the delay signal.")
            .default_value(20.0).scan<'g', double>();

    program.add_argument("--T_f")
            .help("timescale of the feature signal.")
            .default_value(0.1).scan<'g', double>();

    program.add_argument("--Tl_xi")
            .help("learning timescale of xi.")
            .default_value(0.1).scan<'g', double>();

    program.add_argument("--Tl_phi")
            .help("learning timescale of phi.")
            .default_value(0.1).scan<'g', double>();

    program.add_argument("--swindow")
            .help("The smoothing window to remove transients.")
            .default_value(10).scan<'i', int>();

    program.add_argument("--path")
            .default_value(std::string{"tmp/expSM_learning"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
            .help("Location for the result directory");

    program.add_argument("--gpu")
            .help("run code in GPU (WARNING: experiments show CPU is faster)")
            .default_value(false)
            .implicit_value(true);

    program.add_argument("--verbose")
            .help("verbose mode (maximum output)")
            .default_value(false)
            .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    // END argparser section

    // simulation arguments
    const int seed = program.get<int>("--seed");
    auto EXPERIMENT_ID = program.get<std::string>("--experiment_id");
    const int n_neurons = program.get<int>("--n_neurons");
    const double alpha_c = program.get<double>("--alpha_c");
    const double T_d = program.get<double>("--T_d");
    const double T_f = program.get<double>("--T_f");
    const double Tl_xi = program.get<double>("--Tl_xi");
    const double Tl_phi = program.get<double>("--Tl_phi");
    const int diagnostic_timescale = program.get<int>("--diagnostic_timescale");
    const int max_seq_length = program.get<int>("--max_seq_length");
    const int steps_per_memory = program.get<int>("--steps_per_memory");
    const int epochs = program.get<int>("--epochs");
    const int swindow = program.get<int>("--swindow");
    auto path = program.get<std::string>("--path");
    const int simulation_steps = 100000;
    const int diagnostic_size = floor(simulation_steps / diagnostic_timescale);

    // Write out the simulation parameters
    cout<<"Running simulation with "<<std::endl;
    cout<<"simulation steps: "<<simulation_steps<<std::endl;
    cout<<"seed: "<<seed<<std::endl;
    cout<<"n_neurons: "<<n_neurons<<std::endl;
    cout<<"alpha_c: "<<alpha_c<<std::endl;
    cout<<"T_d: "<<T_d<<std::endl;
    cout<<"T_f: "<<T_f<<std::endl;
    cout<<"Tl_xi: "<<Tl_xi<<std::endl;
    cout<<"Tl_phi: "<<Tl_phi<<std::endl;
    cout<<"steps_per_memory: "<<steps_per_memory;
    cout<<"epochs: "<<epochs;
    cout<<"diagnostic timescale: "<< diagnostic_timescale <<std::endl;
    cout<<"max_seq_length: "<< max_seq_length <<std::endl;
    cout<<"sliding window size: "<< swindow <<std::endl;
    cout<<"experiment save path: "<< path <<std::endl;

    auto device = torch::kCPU;
//    if (program["--gpu"] == true) {
//        std::cout << "GPU enabled" << std::endl;
//        device = torch::kCUDA;
//    }
//    else{
//        std::cout<<"device: CPU" << std::endl;
//    }

    auto VERBOSE = program.get<bool>("--verbose");
    // END simulation arguments

    // create directories for writing out results
    // first check if the environment variable for the experiment save directory is specified
    const std::string EXPERIMENT_OUTPUT_DIR = std::getenv("EXPERIMENT_OUTPUT_DIR");
    if(EXPERIMENT_OUTPUT_DIR.empty()) {
        std::cout<<"ERROR: EXPERIMENT_OUTPUT_DIR environment not set";
        return 0;
    }
    fs::path experiment_dir (EXPERIMENT_OUTPUT_DIR);
    fs::path save_path (path);
    fs::path output_path = experiment_dir / save_path;

    fs::create_directories(output_path);

    cout<<"output_path: "<<output_path.string()<<std::endl;
    // END creating directories

    // seed before creating
    torch::manual_seed(seed);

    ExpSM model = ExpSM();
    model.initialize(device, n_neurons, 2*max_seq_length, alpha_c, T_d, T_f, Tl_xi, Tl_phi);
    std::cout<<"Model initialized "<<std::endl;

    std::vector<float> energy_trace;
    if(! model.evaluate_LT_learning(max_seq_length, diagnostic_timescale,
                                    swindow, steps_per_memory, epochs, VERBOSE, energy_trace)){
        std::cout<<"FAIL: no learning";
        return 0;
    }

    // write to file
    fs::path filename = std::string("result_")+EXPERIMENT_ID+std::string(".json");
    fs::path file_save_path = output_path / filename;

    ofstream result_file (file_save_path.string());
    if (result_file.is_open()){
        result_file << "{ \n";
        result_file << "\"seed\": "<<seed<<", \n";
        result_file << "\"alpha_c\": "<<alpha_c<<", \n";
        result_file << "\"n_neurons\": "<<n_neurons<<", \n";
        result_file << "\"T_d\": "<<T_d<<", \n";
        result_file << "\"T_f\": "<<T_f<<", \n";
        result_file << "\"Tl_xi\": "<<Tl_xi<<", \n";
        result_file << "\"Tl_phi\": "<<Tl_phi<<", \n";
        result_file << "\"steps_per_memory\": "<<steps_per_memory<<", \n";
        result_file << "} \n";
        result_file.close();
    } else {
        std::cout<<"ERROR: unable to open file";
    }

    cout<<"File written to "<<file_save_path.string();
    // END write to file

    // write energy to file
    filename = std::string("energy_")+EXPERIMENT_ID+std::string(".txt");
    file_save_path = output_path / filename;

    ofstream energy_file (file_save_path.string());
    if (energy_file.is_open()){
        for(int i=0; i<energy_trace.size(); i++){
            energy_file << energy_trace[i] << std::endl;
        }

        energy_file.close();
    } else {
        std::cout<<"ERROR: unable to open file";
    }

    cout<<"File written to "<<file_save_path.string();
    // END write energy to file

    std::cout<<"SUCCESS: Simulation Complete"<<std::endl;

    return 1;
}
