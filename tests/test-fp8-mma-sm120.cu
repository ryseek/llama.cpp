#include "mma.cuh"

extern "C" __global__ void test_mma_f8_e4m3_sm120(float * output) {
    ggml_cuda_mma::tile<16, 8, float> d = {};
    ggml_cuda_mma::tile<16, 8, int>   a = {};
    ggml_cuda_mma::tile<8, 8, int>    b = {};

    ggml_cuda_mma::mma_f8_e4m3(d, a, b);
    output[threadIdx.x] = d.x[0];
}

int main() {
    return 0;
}
