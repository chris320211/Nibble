#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>


namespace {

__device__ __forceinline__ int decode_int4(uint8_t value) {
    const int four_bit_value = static_cast<int>(value & 0x0F);
    return four_bit_value < 8 ? four_bit_value : four_bit_value - 16;
}


__global__ void int4_linear_kernel_naive(
    const __half* activations,
    const uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    int64_t rows,
    int64_t output_features,
    int64_t input_features,
    int64_t group_size) {
    const int64_t output_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t number_of_outputs = rows * output_features;

    if (output_index >= number_of_outputs) {
        return;
    }

    const int64_t row = output_index / output_features;
    const int64_t output_feature = output_index % output_features;
    const int64_t packed_input_features = input_features / 2;
    const int64_t number_of_groups = input_features / group_size;
    float total = 0.0F;

    for (int64_t group = 0; group < number_of_groups; ++group) {
        const int64_t group_start = group * group_size;
        float group_total = 0.0F;

        for (int64_t offset = 0; offset < group_size; offset += 2) {
            const int64_t input_index = group_start + offset;
            const uint8_t packed = packed_weights[
                output_feature * packed_input_features + input_index / 2];
            const int low_weight = decode_int4(packed);
            const int high_weight = decode_int4(packed >> 4);

            const float low_activation = __half2float(
                activations[row * input_features + input_index]);
            const float high_activation = __half2float(
                activations[row * input_features + input_index + 1]);

            group_total += low_activation * static_cast<float>(low_weight);
            group_total += high_activation * static_cast<float>(high_weight);
        }

        const float scale = __half2float(
            scales[output_feature * number_of_groups + group]);
        total += group_total * scale;
    }

    output[output_index] = __float2half_rn(total);
}


// One warp owns output channel n. Lane t covers 4 consecutive weights
// (2 packed bytes) at k = group * 128 + t * 4.
__global__ void int4_linear_kernel_decode(
    const __half* activations,
    const uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    int64_t output_features,
    int64_t input_features) {
    const int lane = threadIdx.x & 31;
    const int64_t output_feature =
        static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int64_t packed_input_features = input_features / 2;
    const int64_t number_of_groups = input_features / 128;
    const bool in_range = output_feature < output_features;
    const uint8_t* packed_row =
        packed_weights + output_feature * packed_input_features;

    __shared__ __half2 shared_activations[64];
    float total = 0.0F;

    for (int64_t group = 0; group < number_of_groups; ++group) {
        if (threadIdx.x < 64) {
            const __half2* activation_group = reinterpret_cast<const __half2*>(
                activations + group * 128);
            shared_activations[threadIdx.x] = activation_group[threadIdx.x];
        }
        __syncthreads();

        if (in_range) {
            const uint16_t packed_pair = reinterpret_cast<const uint16_t*>(
                packed_row + group * 64)[lane];
            const uint8_t packed_0 = static_cast<uint8_t>(packed_pair);
            const uint8_t packed_1 = static_cast<uint8_t>(packed_pair >> 8);

            const __half2 activation_01 = shared_activations[lane * 2];
            const __half2 activation_23 = shared_activations[lane * 2 + 1];

            float group_dot =
                __low2float(activation_01) *
                    static_cast<float>(decode_int4(packed_0)) +
                __high2float(activation_01) *
                    static_cast<float>(decode_int4(packed_0 >> 4)) +
                __low2float(activation_23) *
                    static_cast<float>(decode_int4(packed_1)) +
                __high2float(activation_23) *
                    static_cast<float>(decode_int4(packed_1 >> 4));

            for (int offset = 16; offset > 0; offset >>= 1) {
                group_dot += __shfl_down_sync(0xffffffff, group_dot, offset);
            }

            if (lane == 0) {
                const float scale = __half2float(
                    scales[output_feature * number_of_groups + group]);
                total += group_dot * scale;
            }
        }

        __syncthreads();
    }

    if (in_range && lane == 0) {
        output[output_feature] = __float2half_rn(total);
    }
}


// Wide-load decode (M = 1). Same warp-per-output-channel layout as
// int4_linear_kernel_decode, but:
// - each lane loads 16 packed bytes (32 weights) with one 128-bit load instead
//   of 2 bytes, so a warp moves 512 bytes per step instead of 64;
// - activations are read straight from global memory (L1-cached and shared by
//   every warp in the block) instead of being staged in shared memory, so there
//   is no __syncthreads anywhere;
// - each lane keeps its own scaled total and the warp reduces once at the end
//   instead of once per group.
// The 32 weights of a lane's chunk always sit inside one 128-weight group, so
// one scale covers the whole chunk. Pointers must be 16-byte aligned.
__global__ void int4_linear_kernel_decode_wide(
    const __half* __restrict__ activations,
    const uint8_t* __restrict__ packed_weights,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int64_t output_features,
    int64_t input_features) {
    const int lane = threadIdx.x & 31;
    const int64_t output_feature =
        static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    // The whole warp shares output_feature, so it exits together and the
    // full-mask shuffles below stay valid.
    if (output_feature >= output_features) {
        return;
    }

    const int64_t number_of_groups = input_features / 128;
    const int64_t number_of_chunks = input_features / 32;
    const uint4* packed_row = reinterpret_cast<const uint4*>(
        packed_weights + output_feature * (input_features / 2));
    const uint4* activation_chunks = reinterpret_cast<const uint4*>(activations);
    const __half* scale_row = scales + output_feature * number_of_groups;

    float total = 0.0F;
#pragma unroll 4
    for (int64_t chunk = lane; chunk < number_of_chunks; chunk += 32) {
        const uint4 packed = packed_row[chunk];
        const uint32_t packed_words[4] = {packed.x, packed.y, packed.z, packed.w};
        float chunk_dot = 0.0F;

        // Packed word w holds weights 8w..8w+7 of the chunk; the matching eight
        // activations are one uint4 (four half2 pairs).
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const uint4 activation_bits = activation_chunks[chunk * 4 + word];
            const __half2* activation_pairs =
                reinterpret_cast<const __half2*>(&activation_bits);
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const uint8_t packed_byte =
                    static_cast<uint8_t>(packed_words[word] >> (8 * byte));
                const __half2 activation_pair = activation_pairs[byte];
                chunk_dot +=
                    __low2float(activation_pair) *
                        static_cast<float>(decode_int4(packed_byte)) +
                    __high2float(activation_pair) *
                        static_cast<float>(decode_int4(packed_byte >> 4));
            }
        }

        total += chunk_dot * __half2float(scale_row[chunk / 4]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (lane == 0) {
        output[output_feature] = __float2half_rn(total);
    }
}


bool is_16_byte_aligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}


// W4A16_DECODE=v1 selects the original shared-memory decode kernel so the two
// can be compared in the same build.
bool use_original_decode_kernel() {
    const char* choice = std::getenv("W4A16_DECODE");
    return choice != nullptr && std::strcmp(choice, "v1") == 0;
}


// Small-batch decode (M = ROWS, 2..8). Same warp-per-output-channel layout as
// int4_linear_kernel_decode, but each warp unpacks its 4 weights once and
// reuses them for every activation row. Each lane applies the group scale to
// its own partial sums, so the warp reduces once per row at the end instead of
// once per group.
template <int ROWS>
__global__ void int4_linear_kernel_small_batch(
    const __half* activations,
    const uint8_t* packed_weights,
    const __half* scales,
    __half* output,
    int64_t output_features,
    int64_t input_features) {
    const int lane = threadIdx.x & 31;
    const int64_t output_feature =
        static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int64_t packed_input_features = input_features / 2;
    const int64_t number_of_groups = input_features / 128;
    const bool in_range = output_feature < output_features;
    const uint8_t* packed_row =
        packed_weights + output_feature * packed_input_features;
    const __half* scale_row = scales + output_feature * number_of_groups;

    __shared__ __half2 shared_activations[ROWS][64];
    float totals[ROWS];
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
        totals[row] = 0.0F;
    }

    for (int64_t group = 0; group < number_of_groups; ++group) {
        for (int index = threadIdx.x; index < ROWS * 64; index += blockDim.x) {
            const int row = index / 64;
            const int column = index % 64;
            const __half2* activation_group = reinterpret_cast<const __half2*>(
                activations + row * input_features + group * 128);
            shared_activations[row][column] = activation_group[column];
        }
        __syncthreads();

        if (in_range) {
            const uint16_t packed_pair = reinterpret_cast<const uint16_t*>(
                packed_row + group * 64)[lane];
            const uint8_t packed_0 = static_cast<uint8_t>(packed_pair);
            const uint8_t packed_1 = static_cast<uint8_t>(packed_pair >> 8);
            const float scale = __half2float(scale_row[group]);
            const float weight_0 = static_cast<float>(decode_int4(packed_0));
            const float weight_1 = static_cast<float>(decode_int4(packed_0 >> 4));
            const float weight_2 = static_cast<float>(decode_int4(packed_1));
            const float weight_3 = static_cast<float>(decode_int4(packed_1 >> 4));

#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
                const __half2 activation_01 = shared_activations[row][lane * 2];
                const __half2 activation_23 = shared_activations[row][lane * 2 + 1];
                const float group_dot =
                    __low2float(activation_01) * weight_0 +
                    __high2float(activation_01) * weight_1 +
                    __low2float(activation_23) * weight_2 +
                    __high2float(activation_23) * weight_3;
                totals[row] += group_dot * scale;
            }
        }

        __syncthreads();
    }

    if (in_range) {
#pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            float total = totals[row];
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            if (lane == 0) {
                output[row * output_features + output_feature] =
                    __float2half_rn(total);
            }
        }
    }
}


template <int ROWS>
void launch_small_batch(
    const at::Tensor& activations,
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    at::Tensor& output,
    int blocks,
    int threads_per_block,
    cudaStream_t stream) {
    int4_linear_kernel_small_batch<ROWS><<<blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
        packed_weights.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        packed_weights.size(0),
        activations.size(1));
}

}  // namespace


at::Tensor int4_linear_cuda(
    const at::Tensor& activations,
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    int64_t group_size) {
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(
        activations.device() == packed_weights.device() &&
            activations.device() == scales.device(),
        "all inputs must be on the same CUDA device");

    TORCH_CHECK(
        activations.scalar_type() == at::kHalf,
        "activations must have dtype torch.float16");
    TORCH_CHECK(
        packed_weights.scalar_type() == at::kByte,
        "packed_weights must have dtype torch.uint8");
    TORCH_CHECK(
        scales.scalar_type() == at::kHalf,
        "scales must have dtype torch.float16");

    TORCH_CHECK(activations.dim() == 2, "activations must have shape [M, K]");
    TORCH_CHECK(
        packed_weights.dim() == 2,
        "packed_weights must have shape [N, K / 2]");
    TORCH_CHECK(scales.dim() == 2, "scales must have shape [N, K / group_size]");
    TORCH_CHECK(activations.is_contiguous(), "activations must be contiguous");
    TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

    TORCH_CHECK(group_size > 0, "group_size must be greater than zero");
    TORCH_CHECK(
        group_size % 2 == 0,
        "group_size must be even because each byte contains two weights");

    const int64_t rows = activations.size(0);
    const int64_t input_features = activations.size(1);
    const int64_t output_features = packed_weights.size(0);

    TORCH_CHECK(
        packed_weights.size(1) * 2 == input_features,
        "packed_weights and activations must describe the same K dimension");
    TORCH_CHECK(
        input_features % group_size == 0,
        "K must be divisible by group_size");
    TORCH_CHECK(
        scales.size(0) == output_features &&
            scales.size(1) == input_features / group_size,
        "scales must have shape [N, K / group_size]");

    const c10::cuda::CUDAGuard device_guard(activations.device());
    at::Tensor output = at::empty(
        {rows, output_features},
        activations.options());

    const int64_t number_of_outputs = rows * output_features;
    if (number_of_outputs == 0) {
        return output;
    }

    constexpr int threads_per_block = 256;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // The warp kernels use vector loads (half2, uint16, uint4). A tensor that
    // starts mid-allocation can be contiguous but misaligned; it takes the
    // naive kernel, which reads one element at a time.
    const bool warp_kernels = group_size == 128 &&
        is_16_byte_aligned(activations.data_ptr()) &&
        is_16_byte_aligned(packed_weights.data_ptr());

    if (rows == 1 && warp_kernels && !use_original_decode_kernel()) {
        constexpr int warps_per_block = 8;
        const int blocks = static_cast<int>(
            (output_features + warps_per_block - 1) / warps_per_block);
        int4_linear_kernel_decode_wide<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
            packed_weights.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            output_features,
            input_features);
    } else if (rows == 1 && warp_kernels) {
        constexpr int warps_per_block = 8;
        const int blocks = static_cast<int>(
            (output_features + warps_per_block - 1) / warps_per_block);
        int4_linear_kernel_decode<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
            packed_weights.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            output_features,
            input_features);
    } else if (rows <= 8 && warp_kernels) {
        constexpr int warps_per_block = 8;
        const int blocks = static_cast<int>(
            (output_features + warps_per_block - 1) / warps_per_block);
        switch (rows) {
            case 2:
                launch_small_batch<2>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            case 3:
                launch_small_batch<3>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            case 4:
                launch_small_batch<4>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            case 5:
                launch_small_batch<5>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            case 6:
                launch_small_batch<6>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            case 7:
                launch_small_batch<7>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
            default:
                launch_small_batch<8>(activations, packed_weights, scales, output,
                                      blocks, threads_per_block, stream);
                break;
        }
    } else {
        const int blocks = static_cast<int>(
            (number_of_outputs + threads_per_block - 1) / threads_per_block);
        int4_linear_kernel_naive<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
            packed_weights.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            rows,
            output_features,
            input_features,
            group_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
