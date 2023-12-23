#include <torch/torch.h>
#include <iostream>
#include "linear.h"

int main() {
    torch::Tensor tensor = torch::rand({2, 3}, torch::device(torch::kCUDA));
    std::cout << tensor << std::endl;

    at::Tensor a = at::ones({2, 2}, torch::device(torch::kCUDA));
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(torch::device(torch::kCUDA));

    std::cout << c << std::endl;
}