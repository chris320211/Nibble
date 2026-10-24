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


__device__ __forceinline__ __half2 bits_to_half2(uint32_t bits) {
    return *reinterpret_cast<const __half2*>(&bits);
}


// Dot product of 8 weights (one packed word; weight j sits in nibble j) with 8
// activations (one uint4 = four half2 pairs). Each nibble goes through
// decode_int4 and an int-to-float conversion.
__device__ __forceinline__ float dot8_decode(
    uint32_t packed_word,
    uint4 activation_bits) {
    const __half2* activation_pairs =
        reinterpret_cast<const __half2*>(&activation_bits);
    float dot = 0.0F;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        const uint8_t packed_byte = static_cast<uint8_t>(packed_word >> (8 * byte));
        const __half2 activation_pair = activation_pairs[byte];
        dot += __low2float(activation_pair) *
                   static_cast<float>(decode_int4(packed_byte)) +
               __high2float(activation_pair) *
                   static_cast<float>(decode_int4(packed_byte >> 4));
    }
    return dot;
}


// Unpacks 8 weights (one packed word) to exact FP16 with no int-to-float
// conversion. FP16 1024.0 is 0x6400 and its mantissa step is exactly 1, so
// OR-ing a 4-bit value u into the low mantissa bits gives the half 1024 + u.
// One LOP3 per pair of weights does three things at once: masks two nibbles
// (one per half), flips their sign bit (two's complement w -> offset binary
// u = w + 8), and sets the 0x6400 exponent. Then:
// - nibbles in bits 0-3 of a half:  (1024 + u) - 1032          = w
// - nibbles in bits 4-7 of a half:  (1024 + 16u) / 16 - 72     = w
// Both are exact in FP16. Pairs come out as weights[0] = (w0, w4),
// weights[1] = (w1, w5), weights[2] = (w2, w6), weights[3] = (w3, w7).
__device__ __forceinline__ void unpack8_fast(
    uint32_t packed_word,
    __half2 (&weights)[4]) {
    constexpr uint32_t low_nibbles = 0x000F000F;
    constexpr uint32_t high_nibbles = 0x00F000F0;
    constexpr uint32_t low_magic = 0x64086408;   // 0x6400 | sign flip at bit 3
    constexpr uint32_t high_magic = 0x64806480;  // 0x6400 | sign flip at bit 7
    const __half2 offset_1032 = bits_to_half2(0x64086408);  // 1032.0
    const __half2 one_sixteenth = bits_to_half2(0x2C002C00);
    const __half2 minus_72 = bits_to_half2(0xD480D480);

    const uint32_t shifted_word = packed_word >> 8;
    weights[0] = __hsub2(
        bits_to_half2((packed_word & low_nibbles) ^ low_magic), offset_1032);
    weights[1] = __hfma2(
        bits_to_half2((packed_word & high_nibbles) ^ high_magic),
        one_sixteenth,
        minus_72);
    weights[2] = __hsub2(
        bits_to_half2((shifted_word & low_nibbles) ^ low_magic), offset_1032);
    weights[3] = __hfma2(
        bits_to_half2((shifted_word & high_nibbles) ^ high_magic),
        one_sixteenth,
        minus_72);
}


// Eight activations (one uint4 = four half2 pairs) as FP32, in order x0..x7.
__device__ __forceinline__ void activations8_to_float(
    uint4 activation_bits,
    float (&activations)[8]) {
    const uint32_t words[4] = {
        activation_bits.x, activation_bits.y, activation_bits.z, activation_bits.w};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        activations[2 * pair] = __low2float(bits_to_half2(words[pair]));
        activations[2 * pair + 1] = __high2float(bits_to_half2(words[pair]));
    }
}


// FP32 dot product of 8 unpacked weights (pair order from unpack8_fast) with
// 8 FP32 activations.
__device__ __forceinline__ float dot8_unpacked(
    const __half2 (&weights)[4],
    const float (&activations)[8]) {
    return activations[0] * __low2float(weights[0]) +
           activations[1] * __low2float(weights[1]) +
           activations[2] * __low2float(weights[2]) +
           activations[3] * __low2float(weights[3]) +
           activations[4] * __high2float(weights[0]) +
           activations[5] * __high2float(weights[1]) +
           activations[6] * __high2float(weights[2]) +
           activations[7] * __high2float(weights[3]);
}


// Same result as dot8_decode with about half the instructions: unpack8_fast
// replaces decode_int4 + int-to-float. The multiply-adds stay in FP32, so the
// result matches the FP32 reference as closely as dot8_decode does.
__device__ __forceinline__ float dot8_fast_unpack(
    uint32_t packed_word,
    uint4 activation_bits) {
    __half2 weights[4];
    float activations[8];
    unpack8_fast(packed_word, weights);
    activations8_to_float(activation_bits, activations);
    return dot8_unpacked(weights, activations);
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
// FAST_UNPACK picks dot8_fast_unpack (v3) over dot8_decode (v2).
// The 32 weights of a lane's chunk always sit inside one 128-weight group, so
// one scale covers the whole chunk. Pointers must be 16-byte aligned.
template <bool FAST_UNPACK>
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
            if constexpr (FAST_UNPACK) {
                chunk_dot += dot8_fast_unpack(packed_words[word], activation_bits);
            } else {
                chunk_dot += dot8_decode(packed_words[word], activation_bits);
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


// Multi-row decode (M = 1). Same 32-weight lane chunks, fast unpack, and single
// final reduction as int4_linear_kernel_decode_wide<true>, but each warp owns
// ROWS_PER_WARP consecutive output channels and blocks hold 4 warps:
// - each activation word is loaded and converted to FP32 once per warp and
//   reused for every row (v3 converts it once per row);
// - each lane has ROWS_PER_WARP weight loads in flight per chunk;
// - each warp does ROWS_PER_WARP times more work per launch and reduction, and
//   smaller blocks spread more evenly over the SMs.
// In the last warp, rows past output_features recompute the last valid row and
// are not stored, so every warp runs the same loop.
template <int ROWS_PER_WARP>
__global__ void int4_linear_kernel_decode_multirow(
    const __half* __restrict__ activations,
    const uint8_t* __restrict__ packed_weights,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int64_t output_features,
    int64_t input_features) {
    const int lane = threadIdx.x & 31;
    const int64_t warp_index =
        static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int64_t first_row = warp_index * ROWS_PER_WARP;
    // The whole warp shares first_row, so it exits together and the full-mask
    // shuffles below stay valid.
    if (first_row >= output_features) {
        return;
    }

    const int64_t number_of_groups = input_features / 128;
    const int64_t number_of_chunks = input_features / 32;
    const uint4* activation_chunks = reinterpret_cast<const uint4*>(activations);
    const uint4* packed_rows[ROWS_PER_WARP];
    const __half* scale_rows[ROWS_PER_WARP];
    float totals[ROWS_PER_WARP];
#pragma unroll
    for (int row = 0; row < ROWS_PER_WARP; ++row) {
        const int64_t output_feature = first_row + row < output_features
            ? first_row + row
            : output_features - 1;
        packed_rows[row] = reinterpret_cast<const uint4*>(
            packed_weights + output_feature * (input_features / 2));
        scale_rows[row] = scales + output_feature * number_of_groups;
        totals[row] = 0.0F;
    }

    // Keep about 4 chunks of weight loads in flight per lane, as v3 does.
#pragma unroll (ROWS_PER_WARP >= 4 ? 1 : 4 / ROWS_PER_WARP)
    for (int64_t chunk = lane; chunk < number_of_chunks; chunk += 32) {
        uint32_t packed_words[ROWS_PER_WARP][4];
        float chunk_dots[ROWS_PER_WARP];
#pragma unroll
        for (int row = 0; row < ROWS_PER_WARP; ++row) {
            const uint4 packed = packed_rows[row][chunk];
            packed_words[row][0] = packed.x;
            packed_words[row][1] = packed.y;
            packed_words[row][2] = packed.z;
            packed_words[row][3] = packed.w;
            chunk_dots[row] = 0.0F;
        }

#pragma unroll
        for (int word = 0; word < 4; ++word) {
            float activation_values[8];
            activations8_to_float(
                activation_chunks[chunk * 4 + word], activation_values);
#pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                __half2 weights[4];
                unpack8_fast(packed_words[row][word], weights);
                chunk_dots[row] += dot8_unpacked(weights, activation_values);
            }
        }

#pragma unroll
        for (int row = 0; row < ROWS_PER_WARP; ++row) {
            totals[row] += chunk_dots[row] * __half2float(scale_rows[row][chunk / 4]);
        }
    }

#pragma unroll
    for (int row = 0; row < ROWS_PER_WARP; ++row) {
        float total = totals[row];
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0 && first_row + row < output_features) {
            output[first_row + row] = __float2half_rn(total);
        }
    }
}


bool is_16_byte_aligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}


// M = 1 decode kernel versions, compared in the same build with W4A16_DECODE:
// v1 = original shared-memory kernel, v2 = wide loads, v3 = wide loads + fast
// unpack, v4 (default) = v3 with several rows per warp.
enum class DecodeVersion { original, wide, fast_unpack, multirow };

DecodeVersion selected_decode_version() {
    const char* choice = std::getenv("W4A16_DECODE");
    if (choice != nullptr && std::strcmp(choice, "v1") == 0) {
        return DecodeVersion::original;
    }
    if (choice != nullptr && std::strcmp(choice, "v2") == 0) {
        return DecodeVersion::wide;
    }
    if (choice != nullptr && std::strcmp(choice, "v3") == 0) {
        return DecodeVersion::fast_unpack;
    }
    return DecodeVersion::multirow;
}


// Output rows per warp for v4, from W4A16_ROWS_PER_WARP (default 4). On a T4,
// 2 and 4 tie within noise across sessions; both beat 1 by 25-35%.
int selected_rows_per_warp() {
    const char* choice = std::getenv("W4A16_ROWS_PER_WARP");
    if (choice == nullptr) {
        return 4;
    }
    const int rows = std::atoi(choice);
    TORCH_CHECK(
        rows == 1 || rows == 2 || rows == 4,
        "W4A16_ROWS_PER_WARP must be 1, 2, or 4");
    return rows;
}


template <int ROWS_PER_WARP>
void launch_decode_multirow(
    const at::Tensor& activations,
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    at::Tensor& output,
    cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    constexpr int rows_per_block = warps_per_block * ROWS_PER_WARP;
    const int64_t output_features = packed_weights.size(0);
    const int blocks = static_cast<int>(
        (output_features + rows_per_block - 1) / rows_per_block);
    int4_linear_kernel_decode_multirow<ROWS_PER_WARP>
        <<<blocks, warps_per_block * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
            packed_weights.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            output_features,
            activations.size(1));
}


template <bool FAST_UNPACK>
void launch_decode_wide(
    const at::Tensor& activations,
    const at::Tensor& packed_weights,
    const at::Tensor& scales,
    at::Tensor& output,
    int blocks,
    int threads_per_block,
    cudaStream_t stream) {
    int4_linear_kernel_decode_wide<FAST_UNPACK>
        <<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
            packed_weights.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            packed_weights.size(0),
            activations.size(1));
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

    if (rows == 1 && warp_kernels) {
        constexpr int warps_per_block = 8;
        const int blocks = static_cast<int>(
            (output_features + warps_per_block - 1) / warps_per_block);
        switch (selected_decode_version()) {
            case DecodeVersion::original:
                int4_linear_kernel_decode<<<blocks, threads_per_block, 0, stream>>>(
                    reinterpret_cast<const __half*>(activations.data_ptr<at::Half>()),
                    packed_weights.data_ptr<uint8_t>(),
                    reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                    output_features,
                    input_features);
                break;
            case DecodeVersion::wide:
                launch_decode_wide<false>(activations, packed_weights, scales, output,
                                          blocks, threads_per_block, stream);
                break;
            case DecodeVersion::fast_unpack:
                launch_decode_wide<true>(activations, packed_weights, scales, output,
                                         blocks, threads_per_block, stream);
                break;
            case DecodeVersion::multirow:
                switch (selected_rows_per_warp()) {
                    case 1:
                        launch_decode_multirow<1>(
                            activations, packed_weights, scales, output, stream);
                        break;
                    case 2:
                        launch_decode_multirow<2>(
                            activations, packed_weights, scales, output, stream);
                        break;
                    default:
                        launch_decode_multirow<4>(
                            activations, packed_weights, scales, output, stream);
                        break;
                }
                break;
        }
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
