#include <torch/extension.h>
#include <vector>

// Forward declaration of the kernel launcher function
void launch_unrolled_convolution(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    const float* filter
);

// The C++ wrapper function that accepts PyTorch Tensors
torch::Tensor forward_convolution(torch::Tensor input, torch::Tensor filter) {
    // Ensure the input tensors are allocated on the GPU and are contiguous in memory
    auto input_gpu = input.device().is_cuda() ? input : input.cuda();
    auto filter_gpu = filter.device().is_cuda() ? filter : filter.cuda();

    auto input_contig = input_gpu.contiguous();
    auto filter_contig = filter_gpu.contiguous();

    int height = input_contig.size(0);
    int width = input_contig.size(1);

    // allocate an empty output tensor on the GPU matching the input size
    auto output = torch::empty_like(input_contig);

    // 3. Extract raw C++ pointers from the PyTorch Tensor objects
    const unsigned char* d_input = (const unsigned char*)input_contig.data_ptr<uint8_t>();
    unsigned char* d_output = (unsigned char*)output.data_ptr<uint8_t>();
    const float* d_filter = filter_contig.data_ptr<float>();

    // launch the optimized CUDA execution pipeline
    launch_unrolled_convolution(d_input, d_output, width, height, d_filter);

    return output;
}

// Bind the C++ function into a Python module named "custom_convolution"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_convolution, "Custom Tiled Coalesced Unrolled 2D Convolution (CUDA)");
}
