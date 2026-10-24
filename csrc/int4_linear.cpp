#include <torch/extension.h>


at::Tensor int4_linear_cuda(
    const at::Tensor& activations,
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    int64_t group_size);

at::Tensor int4_dequantize_cuda(
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    int64_t group_size);


TORCH_LIBRARY(w4a16, m) {
    m.def(
        "linear(Tensor activations, Tensor packed_weights, Tensor scales, "
        "int group_size) -> Tensor");
    m.def(
        "dequantize(Tensor packed_weights, Tensor scales, int group_size) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(w4a16, CUDA, m) {
    m.impl("linear", &int4_linear_cuda);
    m.impl("dequantize", &int4_dequantize_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
