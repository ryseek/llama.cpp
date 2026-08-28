#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __device__ __forceinline__ uint8_t mxfp8_e4m3_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }
    if (!isfinite(amax)) {
        return 0xfe;
    }

    int exponent;
    const float fraction = frexpf(amax, &exponent);
    const int scale_exponent = max(-127, min(127, exponent - 9 + (fraction > 0.875f)));
    return (uint8_t) (scale_exponent + 127);
}

template <typename idx_t>
static __global__ void k_set_rows_f8_scaled(
        const float * src0_ptr,
        const idx_t * src1_ptr,
        uint8_t     * dst_ptr,
        float       * scales_ptr,
        const int32_t head_dim,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int64_t snb0,
        const int64_t snb1,
        const int64_t snb2,
        const int64_t snb3) {
    const float * GGML_CUDA_RESTRICT src0  = src0_ptr;
    uint8_t     * GGML_CUDA_RESTRICT dst    = dst_ptr;
    float       * GGML_CUDA_RESTRICT scales = scales_ptr;

    const int64_t i_head = blockIdx.x;
    const int64_t i01    = blockIdx.y;
    const int64_t i02    = blockIdx.z % ne02;
    const int64_t i03    = blockIdx.z / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = src1_ptr[i10*s10 + i11*s11 + i12*s12];
    ggml_cuda_pdl_lc();

    const float * src_head = src0 + i01*s01 + i02*s02 + i03*s03 + i_head*head_dim;
    uint8_t * dst_head = dst + dst_row*nb1 + i02*nb2 + i03*nb3 + i_head*head_dim;

    float amax = 0.0f;
    for (int32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(src_head[i]));
    }

    __shared__ float reduction[4];
    amax = block_reduce<block_reduce_method::MAX, 128>(amax, reduction);

    __shared__ float inverse_scale;
    if (threadIdx.x == 0) {
        const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
        inverse_scale = 1.0f / scale;
        *(float *) ((char *) scales + i_head*snb0 + dst_row*snb1 + i02*snb2 + i03*snb3) = scale;
    }
    __syncthreads();

    for (int32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const __nv_fp8_e4m3 value(src_head[i] * inverse_scale);
        dst_head[i] = value.__x;
    }
}

template <typename idx_t>
static __global__ void k_set_rows_f8_block_scaled(
        const float * src0_ptr,
        const idx_t * src1_ptr,
        uint8_t     * dst_ptr,
        uint8_t     * scales_ptr,
        const int32_t head_dim,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int64_t snb0,
        const int64_t snb1,
        const int64_t snb2,
        const int64_t snb3) {
    constexpr int block_size = 32;

    const int64_t i_head = blockIdx.x;
    const int64_t i01    = blockIdx.y;
    const int64_t i02    = blockIdx.z % ne02;
    const int64_t i03    = blockIdx.z / ne02;
    const int64_t i10    = i01;
    const int64_t i11    = i02 % ne11;
    const int64_t i12    = i03 % ne12;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = src1_ptr[i10*s10 + i11*s11 + i12*s12];
    ggml_cuda_pdl_lc();

    const int group = threadIdx.x / WARP_SIZE;
    const int dim   = threadIdx.x;
    const int groups_per_head = head_dim / block_size;
    const float * src_head = src0_ptr + i01*s01 + i02*s02 + i03*s03 + i_head*head_dim;
    uint8_t * dst_head = dst_ptr + dst_row*nb1 + i02*nb2 + i03*nb3 + i_head*head_dim;

    const float value = src_head[dim];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
    }

    uint32_t scale_byte = mxfp8_e4m3_scale(amax);
    scale_byte = __shfl_sync(0xffffffff, scale_byte, 0, WARP_SIZE);
    if ((threadIdx.x % WARP_SIZE) == 0) {
        *(uint8_t *) ((char *) scales_ptr + (i_head*groups_per_head + group)*snb0 +
            dst_row*snb1 + i02*snb2 + i03*snb3) = (uint8_t) scale_byte;
    }

    const float scale = ggml_cuda_e8m0_to_fp32((uint8_t) scale_byte);
    const float inverse_scale = amax > 0.0f ? __frcp_rn(scale) : 0.0f;
    const __nv_fp8_e4m3 quantized(value * inverse_scale);
    dst_head[dim] = quantized.__x;
}

template <typename idx_t>
static void set_rows_f8_scaled_cuda(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * scale,
        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t head_dim = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(head_dim == 128 || head_dim == 256);
    GGML_ASSERT(ne00 % head_dim == 0);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->ne[0] == ne00 / head_dim);
    GGML_ASSERT(scale->ne[1] == dst->ne[1]);
    GGML_ASSERT(scale->ne[2] == dst->ne[2]);
    GGML_ASSERT(scale->ne[3] == dst->ne[3]);

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    const dim3 block_size(128);
    const dim3 grid_size(ne00 / head_dim, ne01, ne02*ne03);
    const ggml_cuda_kernel_launch_params launch_params(grid_size, block_size, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_set_rows_f8_scaled<idx_t>, launch_params,
        (const float *) src0->data, (const idx_t *) src1->data, (uint8_t *) dst->data, (float *) scale->data,
        head_dim, ne02, ne11, ne12, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3,
        scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3]);
}

template <typename idx_t>
static void set_rows_f8_block_scaled_cuda(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * scale,
        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t head_dim  = ggml_get_op_params_i32(dst, 0);
    const int32_t block_size = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(head_dim == 128 || head_dim == 256);
    GGML_ASSERT(block_size == 32 && ne00 % head_dim == 0);
    GGML_ASSERT(scale->type == GGML_TYPE_I8);
    GGML_ASSERT(scale->ne[0] == ne00 / block_size);
    GGML_ASSERT(scale->ne[1] == dst->ne[1]);
    GGML_ASSERT(scale->ne[2] == dst->ne[2]);
    GGML_ASSERT(scale->ne[3] == dst->ne[3]);

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    const dim3 block_dim(head_dim);
    const dim3 grid_dim(ne00 / head_dim, ne01, ne02*ne03);
    const ggml_cuda_kernel_launch_params launch_params(grid_dim, block_dim, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_set_rows_f8_block_scaled<idx_t>, launch_params,
        (const float *) src0->data, (const idx_t *) src1->data, (uint8_t *) dst->data, (uint8_t *) scale->data,
        head_dim, ne02, ne11, ne12, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3,
        scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3]);
}

static __global__ void k_set_rows_mxfp8_hot(
        const float * src_ptr,
        const int64_t * logical_ptr,
        half * hot_ptr,
        uint8_t * cold_ptr,
        uint8_t * scale_ptr,
        int32_t head_dim,
        int32_t hot_size,
        int32_t sink_size,
        int64_t cold_size,
        int32_t token_base,
        int64_t src_s1,
        int64_t hot_nb1,
        int64_t cold_nb1,
        int64_t scale_nb0,
        int64_t scale_nb1) {
    constexpr int block_size = 32;

    ggml_cuda_pdl_sync();

    const int token_index = token_base + blockIdx.y;
    const int head        = blockIdx.x;
    const int dim         = threadIdx.x;
    const int group       = dim / block_size;
    const int groups_per_head = head_dim / block_size;
    const int64_t logical = logical_ptr[token_index];
    if (logical < 0 || logical >= hot_size + cold_size) {
        return;
    }
    const int64_t tail_size = hot_size - sink_size;
    const int64_t hot_slot = logical < sink_size ? logical : sink_size + (logical - sink_size) % tail_size;

    half * hot_head = (half *) ((char *) hot_ptr + hot_slot * hot_nb1) + head * head_dim;
    const half old_value = hot_head[dim];

    if (logical >= hot_size) {
        const int64_t cold_row = logical - hot_size;
        const float old_float = __half2float(old_value);
        float amax = fabsf(old_float);
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
        }

        uint32_t scale_byte = mxfp8_e4m3_scale(amax);
        scale_byte = __shfl_sync(0xffffffff, scale_byte, 0, WARP_SIZE);
        if ((threadIdx.x % WARP_SIZE) == 0) {
            *(uint8_t *) ((char *) scale_ptr + (head * groups_per_head + group) * scale_nb0 + cold_row * scale_nb1) =
                (uint8_t) scale_byte;
        }

        const float block_scale = ggml_cuda_e8m0_to_fp32((uint8_t) scale_byte);
        const float inverse_scale = amax > 0.0f ? __frcp_rn(block_scale) : 0.0f;
        const __nv_fp8_e4m3 quantized(old_float * inverse_scale);
        uint8_t * cold_head = cold_ptr + cold_row * cold_nb1 + head * head_dim;
        cold_head[dim] = quantized.__x;
    }

    const float * src_head = src_ptr + token_index * src_s1 + head * head_dim;
    hot_head[dim] = __float2half_rn(src_head[dim]);
}

static void set_rows_mxfp8_hot_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src     = dst->src[0];
    const ggml_tensor * logical = dst->src[1];
    const ggml_tensor * hot     = dst->src[2];
    const ggml_tensor * cold    = dst->src[3];
    const ggml_tensor * scale   = dst->src[4];
    const int32_t head_dim  = ggml_get_op_params_i32(dst, 0);
    const int32_t hot_size  = ggml_get_op_params_i32(dst, 1);
    const int32_t sink_size = ggml_get_op_params_i32(dst, 2);

    GGML_ASSERT(head_dim == 128 || head_dim == 256);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 3) == 32);
    GGML_ASSERT(src->ne[2] == 1 && src->ne[3] == 1 && hot->ne[2] == 1 && hot->ne[3] == 1);
    GGML_ASSERT(cold->ne[2] == 1 && cold->ne[3] == 1 && scale->ne[2] == 1 && scale->ne[3] == 1);
    GGML_ASSERT(src->nb[0] == sizeof(float) && hot->nb[0] == sizeof(half));
    GGML_ASSERT(cold->nb[0] == 1 && scale->nb[0] == 1);
    GGML_ASSERT(src->ne[0] == hot->ne[0] && src->ne[0] == cold->ne[0]);
    GGML_ASSERT(src->ne[0] % head_dim == 0 && hot->ne[1] == hot_size);
    GGML_ASSERT(logical->ne[0] == src->ne[1] && logical->ne[1] == 1 && logical->ne[2] == 1 && logical->ne[3] == 1);
    GGML_ASSERT(logical->nb[0] == sizeof(int64_t));
    GGML_ASSERT(scale->ne[0] == cold->ne[0] / 32 && scale->ne[1] == cold->ne[1]);

    const int32_t n_head = src->ne[0] / head_dim;
    const int32_t tail_size = hot_size - sink_size;
    for (int32_t token_base = 0; token_base < src->ne[1]; token_base += tail_size) {
        const int64_t remaining = src->ne[1] - token_base;
        const int32_t wave_size = remaining < tail_size ? remaining : tail_size;
        const dim3 block_dim(head_dim, 1, 1);
        const dim3 grid_dim(n_head, wave_size, 1);
        const ggml_cuda_kernel_launch_params launch_params(grid_dim, block_dim, 0, ctx.stream());
        ggml_cuda_kernel_launch(k_set_rows_mxfp8_hot, launch_params,
            (const float *) src->data, (const int64_t *) logical->data, (half *) hot->data,
            (uint8_t *) cold->data, (uint8_t *) scale->data, head_dim, hot_size, sink_size, cold->ne[1], token_base,
            src->nb[1] / sizeof(float), hot->nb[1], cold->nb[1], scale->nb[0], scale->nb[1]);
    }
}
#endif

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (dst->type == GGML_TYPE_F16 && dst->src[3] != nullptr && dst->src[3]->type == GGML_TYPE_F8_E4M3) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL);
        set_rows_mxfp8_hot_cuda(ctx, dst);
        return;
#else
        GGML_ABORT("MXFP8 hot-cache writes require CUDA FP8 support");
#endif
    }

    if (dst->type == GGML_TYPE_F8_E4M3) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        GGML_ASSERT(dst->src[3] != nullptr);
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_BLACKWELL);
        const bool block_scaled = dst->src[3]->type == GGML_TYPE_I8;
        if (src1->type == GGML_TYPE_I64) {
            if (block_scaled) {
                set_rows_f8_block_scaled_cuda<int64_t>(ctx, src0, src1, dst->src[3], dst);
            } else {
                set_rows_f8_scaled_cuda<int64_t>(ctx, src0, src1, dst->src[3], dst);
            }
        } else {
            GGML_ASSERT(src1->type == GGML_TYPE_I32);
            if (block_scaled) {
                set_rows_f8_block_scaled_cuda<int32_t>(ctx, src0, src1, dst->src[3], dst);
            } else {
                set_rows_f8_scaled_cuda<int32_t>(ctx, src0, src1, dst->src[3], dst);
            }
        }
        return;
#else
        GGML_ABORT("scaled F8_E4M3 cache writes require CUDA FP8 support");
#endif
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
