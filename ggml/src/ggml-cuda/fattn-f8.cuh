#pragma once

#include "common.cuh"

#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int head_dim>
static __global__ void fattn_f8_to_f16(
        const uint8_t * src_ptr,
        const float *   scale_ptr,
        half *          dst_ptr,
        int64_t         ne1,
        int64_t         ne2,
        int64_t         ne3,
        int64_t         nb1,
        int64_t         nb2,
        int64_t         nb3,
        int64_t         scale_nb0,
        int64_t         scale_nb1,
        int64_t         scale_nb2) {
    constexpr int values_per_block = 2048;
    constexpr int rows_per_block   = values_per_block / head_dim;

    ggml_cuda_pdl_sync();

    const uint8_t * GGML_CUDA_RESTRICT src   = src_ptr;
    const float * GGML_CUDA_RESTRICT   scales = scale_ptr;
    half * GGML_CUDA_RESTRICT          dst    = dst_ptr;

    __shared__ int64_t src_row_offsets[rows_per_block];
    __shared__ float   row_scales[rows_per_block];

    const int64_t nrows    = ne1 * ne2 * ne3;
    const int64_t row_base = (int64_t) blockIdx.x * rows_per_block;

    if (threadIdx.x < rows_per_block) {
        const int64_t row = row_base + threadIdx.x;
        if (row < nrows) {
            const int64_t i3  = row / (ne1 * ne2);
            const int64_t rem = row - i3 * ne1 * ne2;
            const int64_t i2  = rem / ne1;
            const int64_t i1  = rem - i2 * ne1;

            src_row_offsets[threadIdx.x] = i1 * nb1 + i2 * nb2 + i3 * nb3;
            const float row_scale = *(const float *) ((const char *) scales +
                i2 * scale_nb0 + i1 * scale_nb1 + i3 * scale_nb2);
            row_scales[threadIdx.x] = row_scale;
        }
    }
    __syncthreads();

    for (int value = 2 * threadIdx.x; value < values_per_block; value += 2 * blockDim.x) {
        const int row_local = value / head_dim;
        const int64_t row   = row_base + row_local;
        if (row >= nrows) {
            continue;
        }

        const int dim = value - row_local * head_dim;
        __nv_fp8x2_e4m3 packed;
        packed.__x = *(const uint16_t *) (src + src_row_offsets[row_local] + dim);

        const float2 decoded = (float2) packed;
        const float  row_scale = row_scales[row_local];
        ((half2 *) (dst + row * head_dim))[dim / 2] =
            __floats2half2_rn(decoded.x * row_scale, decoded.y * row_scale);
    }
}

template <int head_dim>
static void ggml_cuda_fattn_f8_to_f16_impl(
        const ggml_tensor * src, const ggml_tensor * scale, half * dst, cudaStream_t stream) {
    constexpr int values_per_block = 2048;
    constexpr int rows_per_block   = values_per_block / head_dim;

    const int64_t nrows = ggml_nrows(src);
    const dim3    blocks_num((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3    block_dim(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);

    ggml_cuda_kernel_launch(fattn_f8_to_f16<head_dim>, launch_params,
        (const uint8_t *) src->data, (const float *) scale->data, dst,
        src->ne[1], src->ne[2], src->ne[3], src->nb[1], src->nb[2], src->nb[3],
        scale->nb[0], scale->nb[1], scale->nb[2]);
}

void ggml_cuda_fattn_f8_to_f16(
        const ggml_tensor * src, const ggml_tensor * scale, half * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_BLACKWELL);
    GGML_ASSERT(src->type == GGML_TYPE_F8_E4M3 && src->nb[0] == 1);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && scale->nb[0] == sizeof(float));
    GGML_ASSERT(scale->ne[0] == src->ne[2] && scale->ne[1] == src->ne[1] &&
                scale->ne[2] == src->ne[3] && scale->ne[3] == 1);

    switch (src->ne[0]) {
        case 128:
            ggml_cuda_fattn_f8_to_f16_impl<128>(src, scale, dst, stream);
            break;
        case 256:
            ggml_cuda_fattn_f8_to_f16_impl<256>(src, scale, dst, stream);
            break;
        default:
            GGML_ABORT("unsupported F8_E4M3 Flash Attention head size");
    }
}

template <int head_dim>
static __global__ void fattn_mxfp8_to_f16(
        const uint8_t * src_ptr,
        const uint8_t * scale_ptr,
        half *          dst_ptr,
        int64_t         ne1,
        int64_t         ne2,
        int64_t         ne3,
        int64_t         nb1,
        int64_t         nb2,
        int64_t         nb3,
        int64_t         scale_nb0,
        int64_t         scale_nb1,
        int64_t         scale_nb2) {
    constexpr int block_size       = 32;
    constexpr int values_per_block = 2048;
    constexpr int rows_per_block   = values_per_block / head_dim;
    constexpr int groups_per_head  = head_dim / block_size;

    ggml_cuda_pdl_sync();

    __shared__ int64_t src_row_offsets[rows_per_block];
    __shared__ uint8_t block_scales[rows_per_block * groups_per_head];

    const int64_t nrows    = ne1 * ne2 * ne3;
    const int64_t row_base = (int64_t) blockIdx.x * rows_per_block;

    if (threadIdx.x < rows_per_block) {
        const int64_t row = row_base + threadIdx.x;
        if (row < nrows) {
            const int64_t i3  = row / (ne1 * ne2);
            const int64_t rem = row - i3 * ne1 * ne2;
            const int64_t i2  = rem / ne1;
            const int64_t i1  = rem - i2 * ne1;
            src_row_offsets[threadIdx.x] = i1 * nb1 + i2 * nb2 + i3 * nb3;
        }
    }

    for (int s = threadIdx.x; s < rows_per_block * groups_per_head; s += blockDim.x) {
        const int row_local = s / groups_per_head;
        const int group     = s - row_local * groups_per_head;
        const int64_t row   = row_base + row_local;
        if (row < nrows) {
            const int64_t i3  = row / (ne1 * ne2);
            const int64_t rem = row - i3 * ne1 * ne2;
            const int64_t i2  = rem / ne1;
            const int64_t i1  = rem - i2 * ne1;
            block_scales[s] = *(const uint8_t *) ((const char *) scale_ptr +
                (i2 * groups_per_head + group) * scale_nb0 + i1 * scale_nb1 + i3 * scale_nb2);
        }
    }
    __syncthreads();

    for (int value = 2 * threadIdx.x; value < values_per_block; value += 2 * blockDim.x) {
        const int row_local = value / head_dim;
        const int64_t row   = row_base + row_local;
        if (row >= nrows) {
            continue;
        }

        const int dim   = value - row_local * head_dim;
        const int group = dim / block_size;
        __nv_fp8x2_e4m3 packed;
        packed.__x = *(const uint16_t *) (src_ptr + src_row_offsets[row_local] + dim);

        const float2 decoded = (float2) packed;
        const float scale = ggml_cuda_e8m0_to_fp32(block_scales[row_local * groups_per_head + group]);
        ((half2 *) (dst_ptr + row * head_dim))[dim / 2] =
            __floats2half2_rn(decoded.x * scale, decoded.y * scale);
    }
}

template <int head_dim>
static void ggml_cuda_fattn_mxfp8_to_f16_impl(
        const ggml_tensor * src, const ggml_tensor * scale, half * dst, cudaStream_t stream) {
    constexpr int values_per_block = 2048;
    constexpr int rows_per_block   = values_per_block / head_dim;

    const int64_t nrows = ggml_nrows(src);
    const dim3 blocks_num((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block_dim(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(fattn_mxfp8_to_f16<head_dim>, launch_params,
        (const uint8_t *) src->data, (const uint8_t *) scale->data, dst,
        src->ne[1], src->ne[2], src->ne[3], src->nb[1], src->nb[2], src->nb[3],
        scale->nb[0], scale->nb[1], scale->nb[2]);
}

void ggml_cuda_fattn_mxfp8_to_f16(
        const ggml_tensor * src, const ggml_tensor * scale, half * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_BLACKWELL);
    GGML_ASSERT(src->type == GGML_TYPE_F8_E4M3 && src->nb[0] == 1);
    GGML_ASSERT(scale->type == GGML_TYPE_I8 && scale->nb[0] == 1);
    GGML_ASSERT(src->ne[0] == 128 || src->ne[0] == 256);
    GGML_ASSERT(scale->ne[0] == src->ne[0] / 32 * src->ne[2] && scale->ne[1] == src->ne[1] &&
                scale->ne[2] == src->ne[3] && scale->ne[3] == 1);

    if (src->ne[0] == 128) {
        ggml_cuda_fattn_mxfp8_to_f16_impl<128>(src, scale, dst, stream);
    } else {
        ggml_cuda_fattn_mxfp8_to_f16_impl<256>(src, scale, dst, stream);
    }
}

template <int head_dim>
static __global__ void fattn_mxfp8_decode_cold(
        const uint8_t * cold,
        const uint8_t * scales,
        const int64_t * logical,
        half * dst,
        int64_t n_logical,
        int64_t n_cold,
        int64_t n_heads,
        int64_t n_sequences,
        int64_t cold_nb1,
        int64_t cold_nb2,
        int64_t cold_nb3,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t scale_nb2,
        int32_t hot_size,
        int32_t sink_size,
        int32_t n_kv) {
    constexpr int groups_per_head = head_dim / 32;

    ggml_cuda_pdl_sync();

    const int64_t row      = blockIdx.x;
    const int64_t sequence = row / (n_cold * n_heads);
    const int64_t rem      = row - sequence * n_cold * n_heads;
    const int64_t head     = rem / n_cold;
    const int64_t cold_row = rem - head * n_cold;
    if (sequence >= n_sequences) {
        return;
    }

    const int64_t current_size = logical[n_logical - 1] + 1;
    const int64_t cold_count   = current_size > hot_size ? current_size - hot_size : 0;
    if (cold_row >= cold_count) {
        return;
    }
    const int64_t token = sink_size + cold_row;
    if (token >= n_kv) {
        return;
    }

    __shared__ uint8_t block_scales[groups_per_head];
    if (threadIdx.x < groups_per_head) {
        block_scales[threadIdx.x] = *(const uint8_t *) ((const char *) scales +
            (head * groups_per_head + threadIdx.x) * scale_nb0 + cold_row * scale_nb1 + sequence * scale_nb2);
    }
    __syncthreads();

    const uint8_t * src_row = cold + cold_row * cold_nb1 + head * cold_nb2 + sequence * cold_nb3;
    half * dst_row = dst + ((sequence * n_heads + head) * n_kv + token) * head_dim;
    for (int dim = 2 * threadIdx.x; dim < head_dim; dim += 2 * blockDim.x) {
        __nv_fp8x2_e4m3 packed;
        packed.__x = *(const uint16_t *) (src_row + dim);
        const float2 decoded = (float2) packed;
        const float block_scale = ggml_cuda_e8m0_to_fp32(block_scales[dim / 32]);
        ((half2 *) dst_row)[dim / 2] =
            __floats2half2_rn(decoded.x * block_scale, decoded.y * block_scale);
    }
}

template <int head_dim>
static void ggml_cuda_fattn_mxfp8_decode_cold_impl(
        const ggml_tensor * cold, const ggml_tensor * scale, const ggml_tensor * logical,
        half * dst, int32_t hot_size, int32_t sink_size, int32_t n_kv, cudaStream_t stream) {
    const int64_t max_cold = n_kv > hot_size ? n_kv - hot_size : 0;
    const int64_t n_cold   = max_cold < cold->ne[1] ? max_cold : cold->ne[1];
    if (n_cold == 0) {
        return;
    }

    const dim3 blocks_num(n_cold * cold->ne[2] * cold->ne[3], 1, 1);
    const dim3 block_dim(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(fattn_mxfp8_decode_cold<head_dim>, launch_params,
        (const uint8_t *) cold->data, (const uint8_t *) scale->data, (const int64_t *) logical->data, dst,
        logical->ne[0], n_cold, cold->ne[2], cold->ne[3], cold->nb[1], cold->nb[2], cold->nb[3],
        scale->nb[0], scale->nb[1], scale->nb[2], hot_size, sink_size, n_kv);
}

template <int head_dim>
static __global__ void fattn_mxfp8_overlay_hot(
        const half * hot,
        const int64_t * logical,
        half * dst,
        int64_t n_logical,
        int64_t n_tokens,
        int64_t n_heads,
        int64_t n_sequences,
        int64_t hot_nb1,
        int64_t hot_nb2,
        int64_t hot_nb3,
        int32_t n_hot,
        int32_t hot_size,
        int32_t sink_size) {
    ggml_cuda_pdl_sync();

    const int64_t logical_last = logical[n_logical - 1];
    const int64_t current_size = logical_last + 1;
    const int64_t hot_row      = blockIdx.x;
    const int64_t sequence     = hot_row / (n_hot * n_heads);
    const int64_t rem          = hot_row - sequence * n_hot * n_heads;
    const int64_t head         = rem / n_hot;
    const int64_t local        = rem - head * n_hot;
    if (sequence >= n_sequences) {
        return;
    }

    int64_t token;
    int64_t slot;
    if (local < sink_size) {
        token = local;
        slot  = local;
    } else {
        const int64_t tail_size = hot_size - sink_size;
        const int64_t first_hot = current_size - tail_size > sink_size ? current_size - tail_size : sink_size;
        token = first_hot + local - sink_size;
        if (token >= current_size) {
            return;
        }
        slot = sink_size + (token - sink_size) % tail_size;
    }
    if (token >= current_size || token >= n_tokens) {
        return;
    }

    const half * src_row = (const half *) ((const char *) hot + slot * hot_nb1 + head * hot_nb2 + sequence * hot_nb3);
    half * dst_row = dst + ((sequence * n_heads + head) * n_tokens + token) * head_dim;
    for (int dim = 2 * threadIdx.x; dim < head_dim; dim += 2 * blockDim.x) {
        ((half2 *) dst_row)[dim / 2] = *(const half2 *) (src_row + dim);
    }
}

template <int head_dim>
static void ggml_cuda_fattn_mxfp8_overlay_hot_impl(
        const ggml_tensor * cold, const ggml_tensor * hot, const ggml_tensor * logical,
        half * dst, int32_t hot_size, int32_t sink_size, int32_t n_kv, cudaStream_t stream) {
    const int32_t n_hot = n_kv < hot_size ? n_kv : hot_size;
    const dim3 blocks_num(n_hot * cold->ne[2] * cold->ne[3], 1, 1);
    const dim3 block_dim(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(fattn_mxfp8_overlay_hot<head_dim>, launch_params,
        (const half *) hot->data, (const int64_t *) logical->data, dst, logical->ne[0],
        n_kv, cold->ne[2], cold->ne[3], hot->nb[1], hot->nb[2], hot->nb[3], n_hot, hot_size, sink_size);
}

void ggml_cuda_fattn_mxfp8_hot_to_f16(
        const ggml_tensor * cold, const ggml_tensor * scale, const ggml_tensor * hot,
        const ggml_tensor * logical, half * dst, int32_t hot_size, int32_t sink_size,
        int32_t n_kv, cudaStream_t stream) {
    GGML_ASSERT(cold->type == GGML_TYPE_F8_E4M3 && cold->nb[0] == 1);
    GGML_ASSERT(cold->ne[0] == 128 || cold->ne[0] == 256);
    GGML_ASSERT(scale->type == GGML_TYPE_I8 && scale->nb[0] == 1);
    GGML_ASSERT(scale->ne[0] == cold->ne[0] / 32 * cold->ne[2] && scale->ne[1] == cold->ne[1] &&
                scale->ne[2] == cold->ne[3] && scale->ne[3] == 1);
    GGML_ASSERT(hot->type == GGML_TYPE_F16 && hot->ne[0] == cold->ne[0]);
    GGML_ASSERT(hot->ne[1] == hot_size && hot->ne[2] == cold->ne[2] && hot->ne[3] == cold->ne[3]);
    GGML_ASSERT(hot->nb[0] == sizeof(half));
    GGML_ASSERT(logical->type == GGML_TYPE_I64 && logical->ne[0] > 0 && logical->nb[0] == sizeof(int64_t));
    GGML_ASSERT(hot_size > sink_size && sink_size >= 0);
    GGML_ASSERT(n_kv > 0 && n_kv <= cold->ne[1] + hot_size);
    CUDA_CHECK(cudaMemsetAsync(dst, 0, cold->ne[0] * n_kv * cold->ne[2] * cold->ne[3] * sizeof(half), stream));
    if (cold->ne[0] == 128) {
        ggml_cuda_fattn_mxfp8_decode_cold_impl<128>(cold, scale, logical, dst, hot_size, sink_size, n_kv, stream);
        ggml_cuda_fattn_mxfp8_overlay_hot_impl<128>(cold, hot, logical, dst, hot_size, sink_size, n_kv, stream);
    } else {
        ggml_cuda_fattn_mxfp8_decode_cold_impl<256>(cold, scale, logical, dst, hot_size, sink_size, n_kv, stream);
        ggml_cuda_fattn_mxfp8_overlay_hot_impl<256>(cold, hot, logical, dst, hot_size, sink_size, n_kv, stream);
    }
}

template <int head_dim>
static __global__ void flash_attn_ext_f8_ref(const float *   Q_ptr,
                                             const uint8_t * K_ptr,
                                             const uint8_t * V_ptr,
                                             const half *    mask_ptr,
                                             const float *   sinks_ptr,
                                             const float *   k_scale_ptr,
                                             const float *   v_scale_ptr,
                                             float *         dst_ptr,
                                             float           scale,
                                             float           max_bias,
                                             float           m0,
                                             float           m1,
                                             uint32_t        n_head_log2,
                                             float           logit_softcap,
                                             int32_t         n_kv,
                                             int32_t         n_head,
                                             int32_t         n_head_kv,
                                             int32_t         n_queries,
                                             int64_t         q_nb1,
                                             int64_t         q_nb2,
                                             int64_t         q_nb3,
                                             int64_t         k_nb1,
                                             int64_t         k_nb2,
                                             int64_t         k_nb3,
                                             int64_t         v_nb1,
                                             int64_t         v_nb2,
                                             int64_t         v_nb3,
                                             int64_t         mask_nb1,
                                             int64_t         mask_nb3,
                                             int32_t         mask_ne3,
                                             int64_t         ks_nb0,
                                             int64_t         ks_nb1,
                                             int64_t         ks_nb2,
                                             int64_t         vs_nb0,
                                             int64_t         vs_nb1,
                                             int64_t         vs_nb2) {
    ggml_cuda_pdl_sync();

    const float * GGML_CUDA_RESTRICT   Q       = Q_ptr;
    const uint8_t * GGML_CUDA_RESTRICT K       = K_ptr;
    const uint8_t * GGML_CUDA_RESTRICT V       = V_ptr;
    const half * GGML_CUDA_RESTRICT    mask    = mask_ptr;
    const float * GGML_CUDA_RESTRICT   sinks   = sinks_ptr;
    const float * GGML_CUDA_RESTRICT   k_scale = k_scale_ptr;
    const float * GGML_CUDA_RESTRICT   v_scale = v_scale_ptr;
    float * GGML_CUDA_RESTRICT         dst     = dst_ptr;

    const int32_t query    = blockIdx.x;
    const int32_t head     = blockIdx.y;
    const int32_t sequence = blockIdx.z;
    const int32_t kv_head  = head / (n_head / n_head_kv);
    const int32_t dim      = threadIdx.x;

    const float * q = (const float *) ((const char *) Q + query * q_nb1 + head * q_nb2 + sequence * q_nb3);
    const half *  mask_row =
        mask ? (const half *) ((const char *) mask + query * mask_nb1 + (sequence % mask_ne3) * mask_nb3) : nullptr;

    float value_acc = 0.0f;
    float row_sum   = 0.0f;
    float row_max   = -FLT_MAX;

    __shared__ float reduction[head_dim / WARP_SIZE];
    __shared__ float logit;

    for (int32_t token = 0; token < n_kv; ++token) {
        const uint8_t *     k       = K + token * k_nb1 + kv_head * k_nb2 + sequence * k_nb3;
        const __nv_fp8_e4m3 k_value = *(const __nv_fp8_e4m3 *) (k + dim);

        float partial = q[dim] * (float) k_value;
        partial       = block_reduce<block_reduce_method::SUM, head_dim>(partial, reduction);

        if (dim == 0) {
            const float ks =
                *(const float *) ((const char *) k_scale + kv_head * ks_nb0 + token * ks_nb1 + sequence * ks_nb2);
            float dot = partial * ks * scale;
            if (logit_softcap != 0.0f) {
                dot = logit_softcap * tanhf(dot / logit_softcap);
            }
            if (mask_row) {
                dot += get_alibi_slope(max_bias, head, n_head_log2, m0, m1) * __half2float(mask_row[token]);
            }
            logit = dot;
        }
        __syncthreads();

        const float row_max_new = fmaxf(row_max, logit);
        const float old_scale   = expf(row_max - row_max_new);
        const float weight      = expf(logit - row_max_new);

        const uint8_t *     v       = V + token * v_nb1 + kv_head * v_nb2 + sequence * v_nb3;
        const __nv_fp8_e4m3 v_value = *(const __nv_fp8_e4m3 *) (v + dim);
        const float         vs =
            *(const float *) ((const char *) v_scale + kv_head * vs_nb0 + token * vs_nb1 + sequence * vs_nb2);

        value_acc = value_acc * old_scale + weight * (float) v_value * vs;
        row_sum   = row_sum * old_scale + weight;
        row_max   = row_max_new;
        __syncthreads();
    }

    if (sinks) {
        const float sink        = sinks[head];
        const float row_max_new = fmaxf(row_max, sink);
        const float old_scale   = expf(row_max - row_max_new);
        const float sink_weight = expf(sink - row_max_new);
        value_acc *= old_scale;
        row_sum = row_sum * old_scale + sink_weight;
    }

    dst[((int64_t) sequence * n_queries * n_head + (int64_t) query * n_head + head) * head_dim + dim] =
        value_acc / row_sum;
}

template <int head_dim>
static void ggml_cuda_flash_attn_ext_f8_ref_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sinks   = dst->src[4];
    const ggml_tensor * k_scale = dst->src[5];
    const ggml_tensor * v_scale = dst->src[6];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F8_E4M3 && V->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(Q->ne[0] == head_dim && K->ne[0] == head_dim && V->ne[0] == head_dim);
    GGML_ASSERT(k_scale && v_scale);

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const dim3                           block_size(head_dim);
    const dim3                           grid_size(Q->ne[1], Q->ne[2], Q->ne[3]);
    const ggml_cuda_kernel_launch_params launch_params(grid_size, block_size, 0, ctx.stream());
    ggml_cuda_kernel_launch(flash_attn_ext_f8_ref<head_dim>, launch_params, (const float *) Q->data,
                            (const uint8_t *) K->data, (const uint8_t *) V->data,
                            mask ? (const half *) mask->data : nullptr, sinks ? (const float *) sinks->data : nullptr,
                            (const float *) k_scale->data, (const float *) v_scale->data, (float *) dst->data, scale,
                            max_bias, m0, m1, n_head_log2, logit_softcap, K->ne[1], Q->ne[2], K->ne[2], Q->ne[1],
                            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
                            mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, k_scale->nb[0],
                            k_scale->nb[1], k_scale->nb[2], v_scale->nb[0], v_scale->nb[1], v_scale->nb[2]);
}

static void ggml_cuda_flash_attn_ext_f8_ref(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[0]) {
        case 128:
            ggml_cuda_flash_attn_ext_f8_ref_impl<128>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_f8_ref_impl<256>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported F8_E4M3 Flash Attention head size");
    }
}

#else

void ggml_cuda_fattn_f8_to_f16(
        const ggml_tensor *, const ggml_tensor *, half *, cudaStream_t) {
    GGML_ABORT("F8_E4M3 Flash Attention requires CUDA FP8 support");
}

void ggml_cuda_fattn_mxfp8_to_f16(
        const ggml_tensor *, const ggml_tensor *, half *, cudaStream_t) {
    GGML_ABORT("MXFP8 Flash Attention requires CUDA FP8 support");
}

void ggml_cuda_fattn_mxfp8_hot_to_f16(
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, half *,
        int32_t, int32_t, int32_t, cudaStream_t) {
    GGML_ABORT("MXFP8 hot-cache Flash Attention requires CUDA FP8 support");
}

static void ggml_cuda_flash_attn_ext_f8_ref(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("F8_E4M3 Flash Attention requires CUDA FP8 support");
}

#endif
